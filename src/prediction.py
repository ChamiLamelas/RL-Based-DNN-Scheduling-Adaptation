#!/usr/bin/env python3.8

import torch
from tqdm import tqdm
import gpu


def forward(model, data, eval=True, dev=gpu.get_device()):
    model, data = gpu.move(dev, model, data)
    if eval:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        return model(data)


def predict(model, data_loader, dev=gpu.get_device()):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(
            data_loader, total=len(data_loader), desc=f"prediction epoch"
        ):
            data, target = gpu.move(dev, *batch)
            correct += num_correct(model(data), target)
            total += data.size()[0]
    return correct / total


def num_correct(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().sum().item()
