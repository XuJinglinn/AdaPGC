import torch
from utilities import accuracy

def evaluate_model(tta_model, dataloader, device, args):
    tta_model.eval()
    accs = []
    data_bar = dataloader
    iters = len(data_bar)

    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(data_bar):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)
            
            outputs, loss = tta_model((a_input, v_input), adapt_flag=False)
            acc = accuracy(outputs[1], labels, topk=(1,))
            accs.append(acc[0].item())
    return round(sum(accs) / len(accs), 2)
