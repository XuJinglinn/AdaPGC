# AdaPGC

### Multi-modal Test-time Adaptation via Adaptive Probabilistic Gaussian Calibration

**CVPR 2026 · Official Implementation**

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Multi-modal_Test-time_Adaptation_via_Adaptive_Probabilistic_Gaussian_Calibration_CVPR_2026_paper.html">
    <img src="https://img.shields.io/badge/CVPR%202026-Paper-537895?style=flat" alt="CVPR 2026 Paper">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.19093">
    <img src="https://img.shields.io/badge/arXiv-2604.19093-B65C5C?style=flat" alt="arXiv">
  </a>
</p>


## Overview

AdaPGC adapts multi-modal models using unlabeled test data. It combines explicit class-conditional Gaussian modeling with adaptive contrastive rectification to address modality asymmetry and calibrate predictions under distribution shifts. See the [paper](https://arxiv.org/abs/2604.19093) for the method.

This repository provides the adaptation implementation and evaluation entry point built on a fine-tuned CAV-MAE model.

The code supports **Kinetics50** and **VGGSound**, with separate audio and video corruption settings. 

## Getting Started

### 1. Prepare the workspace

```bash
git clone https://github.com/XuJinglinn/AdaPGC.git
cd AdaPGC
```

Use a CUDA-enabled PyTorch environment. The current entry point runs the model on GPU; the launch script uses Bash.

The model and data loader depend on `torch`, `torchaudio`, `torchvision`, `timm`, `numpy`, and `Pillow`. The adaptation module also imports `scikit-learn` and `torch-ema`; the runner uses `tqdm`. This list describes the inspected imports, not a version-pinned environment specification.

### 2. Prepare data and a checkpoint

Follow the [READ benchmark](https://github.com/XLearning-SCU/2024-ICLR-READ) for data preparation. Prepare the corruption metadata, class-label CSV, and a matching fine-tuned CAV-MAE checkpoint.

Set the local paths and GPU IDs in [`script/run_adapgc.sh`](script/run_adapgc.sh):

| Argument                                           | Supplied value                                 |
| :------------------------------------------------- | :--------------------------------------------- |
| `--json-root`                                      | `json_csv_files/ks50/`                         |
| `--label-csv`                                      | `json_csv_files/class_labels_indices_ks50.csv` |
| `--pretrain_path` | `checkpoints/cav_mae_ks50.pth` |                                                |
| `--gpu`                                            | `0,1`                                          |

These are paths expected by the script; prepare the corresponding files before running. For a single GPU, change `--gpu` to `0`.

The runner resolves corruption metadata relative to `--json-root`:

```text
<json-root>/
├── audio/
│   └── <corruption>/
│       └── severity_<level>.json
├── video/
│   └── <corruption>/
│       └── severity_<level>.json
└── clean/
    └── severity_0.json
```

Each JSON file contains a top-level `data` list. Standard entries use `wav`, `labels`, `video_id`, and `video_path`; the label CSV uses `mid`, `index`, and `display_name`. Ensure that the referenced audio and frame paths are valid on your machine.

### 3. Run adaptation

From the repository root:

```bash
bash script/run_adapgc.sh
```

Logs are written beneath `exp_logs/` by default. Use `--log-dir` and `--exp-name` to organize runs.

The supplied script sets batch size to `16`, `--w-c` to `0.01`, `--w-g` to `1`, and `--gamma` to `1`.

To customize a run, edit the script or invoke [`run_adapgc.py`](run_adapgc.py) directly. Available controls include `--dataset`, `--corruption-modality`, `--severity-start`, and `--severity-end`. When changing datasets, also update the metadata, labels, and checkpoint paths.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xu2026adapgc,
  title     = {Multi-modal Test-time Adaptation via Adaptive Probabilistic Gaussian Calibration},
  author    = {Xu, Jinglin and Li, Yi and Sun, Chuxiong and Xu, Xiao and Li, Jiangmeng and Xu, Fanjiang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  url       = {https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Multi-modal_Test-time_Adaptation_via_Adaptive_Probabilistic_Gaussian_Calibration_CVPR_2026_paper.html}
}
```

## Acknowledgements

This repository builds on [READ](https://github.com/XLearning-SCU/2024-ICLR-READ) and [CAV-MAE](https://github.com/yuangongnd/cav-mae). We thank their authors for making the code available.
