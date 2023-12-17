#!/usr/bin/env python3.8

import torch
from torchvision import datasets, transforms
import os

DATA_FOLDER = os.path.join("..", "data")


def cifar10(train, batch_size):
    def _cifar10_load(download):
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(
                DATA_FOLDER,
                train=train,
                download=download,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=train,
            pin_memory=True,
        )

    try:
        return _cifar10_load(False)
    except RuntimeError:
        return _cifar10_load(True)


if __name__ == "__main__":
    dataset = cifar10(True, 1)
    element = next(iter(dataset))[0]
    print(element.requires_grad)
