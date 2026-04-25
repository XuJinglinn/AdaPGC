import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_ema import ExponentialMovingAverage
import torch
import torch, gc

def sym_kl_from_logits(z1: torch.Tensor, z2: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Calculate the symmetric KL divergence (batch form) for two sets of logits.
    """
    p = F.softmax(z1, dim=dim).clamp_min(eps)
    q = F.softmax(z2, dim=dim).clamp_min(eps)

    kl_pq = (p * (p.log() - q.log())).sum(dim=dim)
    kl_qp = (q * (q.log() - p.log())).sum(dim=dim)
    return 0.5 * (kl_pq + kl_qp)


class IncrementalGMM:
    """
    Incremental Gaussian Mixture Model with soft-label batch updates and robust Cholesky-based prediction.
    Maintains sufficient statistics (Nk, Sk, Qk) for each component and ensures positive-definite covariances.
    """
    def __init__(self, n_components, n_features, init_mu, init_bias, alpha=0.9, reg_covar=1e-6, dtype=torch.float32, device=None):
        self.K = n_components
        self.d = n_features
        self.alpha = alpha
        self.reg_covar = reg_covar
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        # Initialize parameters
        self.weights_ = torch.ones(self.K, dtype=dtype, device=self.device) / self.K
        
        if init_mu is not None:
            init_mu = init_mu.to(dtype=self.dtype, device=self.device)  # [K, d-1]
            init_bias = init_bias.to(dtype=self.dtype, device=self.device)  # [K]
            assert init_mu.shape == (self.K, self.d), f"init_mu must have shape ({self.K}, {self.d})"
            assert init_bias is not None and init_bias.shape == (self.K,), f"init_bias must have shape ({self.K},)"
            self.means_ = init_mu.to(dtype=self.dtype, device=self.device)
        else:
            assert self.d > 0, "Feature dimension must be positive"
            self.means_ = torch.zeros(self.K, self.d, dtype=dtype, device=self.device)
        eye = torch.eye(self.d, dtype=dtype, device=self.device)
        self.covariances_ = eye.unsqueeze(0).repeat(self.K, 1, 1)
        # Compute robust Cholesky factors
        self._update_cholesky()
        # initialize weights
        self.weights_ = torch.exp(init_bias + 0.5 * self.log_det_cov_ + 0.5 * self.means_.pow(2).sum(dim=1))  # [K]
        
        N_pseudo = 0  # pseudo-samples for each component
        self.Nk = torch.ones(self.K, dtype=dtype, device=self.device) * N_pseudo
        self.Sk = self.Nk.unsqueeze(1) * self.means_
        self.Qk = torch.einsum('k, kij -> kij', self.Nk, torch.eye(self.d, dtype=dtype, device=self.device).unsqueeze(0).repeat(self.K,1,1))
        self.Qk += torch.einsum('k, ki, kj -> kij', self.Nk, self.means_, self.means_)
        self.N = self.Nk.sum()
        
    def _update_cholesky(self):
        """
        Compute Cholesky decomposition with jitter to ensure positive-definiteness.
        """
        cov = self.covariances_.clone()
        jitter = self.reg_covar
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(cov)
                # success
                self.cholesky_L = L
                diag = torch.diagonal(L, dim1=-2, dim2=-1)
                self.log_det_cov_ = 2.0 * torch.sum(torch.log(diag), dim=-1)
                return
            except RuntimeError:
                cov = cov + jitter * torch.eye(self.d, dtype=self.dtype, device=self.device)
                jitter *= 10
        raise RuntimeError("Covariance not positive-definite even after adding jitter")

    def update_batch(self, X, gamma):
        with torch.no_grad():
            """Incremental update with batch X and responsibilities gamma."""
            X = X.to(dtype=self.dtype, device=self.device)
            gamma = gamma.to(dtype=self.dtype, device=self.device)
            B, d = X.shape
            assert d == self.d, f"Feature dimension mismatch: got {d}, expected {self.d}"
            assert gamma.shape == (B, self.K), f"Gamma must have shape (B, {self.K})"

            # Update sufficient statistics
            alpha = self.alpha
            delta_Nk = gamma.sum(dim=0)
            delta_Sk = gamma.t() @ X
            delta_Qk = torch.einsum('bk,bi,bj->kij', gamma, X, X)
            self.Nk += delta_Nk
            self.Sk += delta_Sk
            self.Qk += delta_Qk
            self.N += delta_Nk.sum()
            # Recompute parameters
            means = self.means_
            self.weights_ = alpha * self.weights_ + (1 - alpha) * self.Nk / self.N
            self.means_ = alpha * self.means_ + (1 - alpha) * self.Sk / self.Nk.unsqueeze(1)
            
            alpha = 0.9
            means = self.means_
            for k in range(self.K):
                self.covariances_[k] = alpha * self.covariances_[k] + (1 - alpha) * (
                    ((self.covariances_[k] + torch.outer(means[k], means[k])) * (self.Nk[k]) + delta_Qk[k])/ self.Nk[k] - torch.outer(self.means_[k], self.means_[k])  # [d, d]
                )
            self._update_cholesky()

    def predict_batch(self, X, device):
        """Return logits for batch X using Cholesky solves."""
        X = X.to(dtype=self.dtype, device=self.device)
        B, d = X.shape
        const = -0.5 * (d * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device)))
        logits = torch.zeros(B, self.K, dtype=self.dtype, device=self.device)
        
        mu = self.means_.unsqueeze(0).expand(B, -1, -1)  # [B, K, d]

        # X: [B, d] -> [B, 1, d] -> broadcast to [B, K, d]
        X_expand = X.unsqueeze(1).expand(-1, self.K, -1)      # [B, K, d]

        # Convert to column vectors [K, d, 1]
        x_vec = X_expand.unsqueeze(-1)                       # [K, d, 1]
        mu_vec = mu.unsqueeze(-1)                            # [K, d, 1]

        # Σ^{-1} x and Σ^{-1} μ, both with shape [K, d, 1]
        Sigma_inv_x = torch.cholesky_solve(x_vec, self.cholesky_L)         # [K, d, 1]
        Sigma_inv_mu = torch.cholesky_solve(mu_vec, self.cholesky_L)       # [K, d, 1]

        # Compute the three quadratic terms
        xx   = torch.matmul(x_vec.transpose(-2, -1), Sigma_inv_x).squeeze(-1).squeeze(-1)     # [K]
        xmu  = torch.matmul(x_vec.transpose(-2, -1), Sigma_inv_mu).squeeze(-1).squeeze(-1)    # [K]
        mumu = torch.matmul(mu_vec.transpose(-2, -1), Sigma_inv_mu).squeeze(-1).squeeze(-1)   # [K]

        # Squared Mahalanobis distance [K]
        maha =  xx - 2 * xmu + mumu
        
        logits = torch.log(self.weights_) - 0.5 * self.log_det_cov_ - 0.5 * maha + const  # [B, K]
        logits = logits.to(device) if device is not None else logits
        return logits
        
    def get_params(self):
        """Return weights, means, covariances."""
        return self.weights_, self.means_, self.covariances_
    
    def dispose(self):
        """Release GPU/CPU memory occupied by this GMM (safe to call multiple times)."""
        # 1) If on GPU, synchronize this device first
        try:
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
        except Exception:
            pass

        # 2) Set attributes to None (break Python reference chain so tensors can be freed)
        for name in [
            'Nk','Sk','Qk','weights_','means_','covariances_',
            'cholesky_L','log_det_cov_'
        ]:
            if hasattr(self, name):
                try:
                    setattr(self, name, None)
                except Exception:
                    pass

        # 3) Force garbage collection
        gc.collect()

        # 4) Clear CUDA caches on this device
        try:
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass



class ADAPGC(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device
        init_mu = model.module.mlp_head[-1].weight.detach().clone()  # [K, D]
        init_bias = model.module.mlp_head[-1].bias.detach().clone()  # [K]
        
        gmm_device   = torch.device(f"cuda:1")
        device = gmm_device
        self.gmm_f = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias,reg_covar=1e-6, device=device, dtype=torch.float32)
        self.gmm_a = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias,reg_covar=1e-6, device=device, dtype=torch.float32)
        self.gmm_v = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias,reg_covar=1e-6, device=device, dtype=torch.float32)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        ln_params = collect_ln_params(self.model)
        print('requires_grad:', sum(p.requires_grad for p in ln_params))
        print('len:', len(ln_params))
        self.ln_optimizer = torch.optim.Adam([{'params': ln_params, 'lr': 1e-4}], weight_decay=0., betas=(0.9, 0.999))

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, self.ema, self.gmm_f, self.gmm_a, self.gmm_v, self.ln_optimizer)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                loss = (0, 0)
                outputs = (outputs, outputs)

        return outputs, loss
    
    def delete_gmm(self):
        """Release GPU/CPU memory occupied by all GMM instances."""
        self.gmm_f.dispose()
        self.gmm_a.dispose()
        self.gmm_v.dispose()


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler, ema, gmm_f, gmm_a, gmm_v, ln_optimizer):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    with autocast():
        # forward
        results = model.module.forward_eval_with_features(a=x[0], v=x[1])
        outputs = results['logits']
        ca = results['ca']
        cv = results['cv']
        feat = results['feat']
        mask = results['mask']

    p_sum = outputs.softmax(dim=-1).sum(dim=0)
    loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()    

    pred = outputs.softmax(dim=-1)
    pred_max = pred.max(dim=-1)[0]
    gamma = math.exp(-1)
    t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
    loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()

    feat_full = feat[mask['full']]
    gmm_f.update_batch(feat, pred)
    gmm_a.update_batch(ca, pred)
    gmm_v.update_batch(cv, pred)
    
    logits_gmm = gmm_f.predict_batch(feat_full, device=pred.device)
    logits_a = gmm_a.predict_batch(ca, device=pred.device)
    logits_v = gmm_v.predict_batch(cv, device=pred.device)
    D_a = sym_kl_from_logits(logits_gmm, logits_a)  # [N]
    D_v = sym_kl_from_logits(logits_gmm, logits_v)  # [N]
    
    loss_c = torch.tensor(0.0, device=outputs.device)
    # If audio is more reliable: pull v → a; if video is more reliable: pull a → v
    mask_v2a = (D_a < D_v)       # audio more reliable, pull v toward a
    mask_a2v = (D_v <= D_a)      # video more reliable, pull a toward v
    
    tau = 0.05
    a_f = F.normalize(ca, p=2, dim=-1) 
    v_f = F.normalize(cv, p=2, dim=-1)
    v_f_t = v_f.detach()
    a_f_t = a_f.detach()
    
    idx_v2a = mask_v2a.nonzero(as_tuple=False).squeeze(-1)  # v->a
    idx_a2v = mask_a2v.nonzero(as_tuple=False).squeeze(-1)  # a->v

    loss_parts = []
    num_terms = 0
    
    # v -> a: anchor is v_f[idx], positive samples are at the same indices in a_f_t
    if idx_v2a.numel() > 0:
        logits_va = v_f.index_select(0, idx_v2a) @ a_f_t.t() / tau      # [n1, N]
        labels_va = idx_v2a                                             # positive sample index = original index
        loss_va = F.cross_entropy(logits_va, labels_va, reduction='sum')
        loss_parts.append(loss_va)
        num_terms += idx_v2a.numel()

    # a -> v: anchor is a_f[idx], positive samples are at the same indices in v_f_t
    if idx_a2v.numel() > 0:
        logits_av = a_f.index_select(0, idx_a2v) @ v_f_t.t() / tau      # [n2, N]
        labels_av = idx_a2v
        loss_av = F.cross_entropy(logits_av, labels_av, reduction='sum')
        loss_parts.append(loss_av)
        num_terms += idx_a2v.numel()

    # Aggregate
    if num_terms > 0:
        loss_c = torch.stack(loss_parts).sum() / num_terms
    else:
        loss_c = torch.tensor(0.0, device=outputs.device, dtype=ca.dtype)

    p = logits_gmm.softmax(dim=1)
    log_q = torch.log_softmax(outputs, dim=1)
    loss_gmm = -(p * log_q).sum(dim=1).mean()

    loss =  loss_ra  - loss_bal + args.w_g * loss_gmm
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)

    # Update ln_optimizer using only loss_c
    loss_c = args.w_c * loss_c
    ln_optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss_c).backward()
    scaler.step(ln_optimizer)

    scaler.update()

    ema.update(model.parameters())
    
    ema.store()
    ema.copy_to(model.parameters())

    with torch.no_grad():
        with autocast():
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)
            outputs2 += args.gamma * logits_gmm 
    ema.restore()

    return (outputs, outputs2), (loss_ra.item(), loss_bal.item())


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion_qkv = []
    names_fusion_qkv = []

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv


def collect_ln_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)

    return model
