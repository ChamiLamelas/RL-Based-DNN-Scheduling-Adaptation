#!/usr/bin/env python3.8

import torch


def move(device, *objs):
    return [obj.to(device) for obj in objs]


def get_device(device=0):
    if torch.cuda.is_available():
        if 0 <= device < torch.cuda.device_count():
            return torch.device(f"cuda:{device}")
        raise RuntimeError(
            f"You specified a GPU that is not available ({device}), machine has {torch.cuda.device_count()} GPUs"
        )
    return torch.device("cpu")
