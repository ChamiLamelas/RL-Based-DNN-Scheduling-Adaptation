#!/usr/bin/env python3.8

import torch.nn as nn
import torch

DEEPEN_BLOCK_NAME = "Net2NetDeepenBlock"


def is_deepen_block(layer):
    return DEEPEN_BLOCK_NAME in type(layer).__name__ and layer.original


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters())


def get_str_rep(layer):
    if layer is None:
        return "none-layer"
    elif is_deepen_block(layer):
        layer = layer.layers[0]
    return str(layer)


class FeedForwardNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())
        self.original = True

    def forward(self, x):
        return self.layers(x)


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = FeedForwardNet2NetDeepenBlock(3072, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class FeedForwardNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FeedForwardNet2NetDeepenBlock(3072, 32)
        self.fc2 = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FeedForwardNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FeedForwardNet2NetDeepenBlock(3072, 32)
        self.fc2 = FeedForwardNet2NetDeepenBlock(32, 32)
        self.fc3 = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ConvolutionalNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs), nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class NormalizedConvolutionalNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.original = True

    def forward(self, x):
        return self.layers(x)


class BatchNormConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(in_channels, 32, 3)
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(32, 64, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(3, 32, 3)
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3)
        self.pool1 = nn.AvgPool2d((2, 2))
        self.conv3 = NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(3, 32, 3)
        self.pool1 = nn.AvgPool2d((2, 2))
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(3, 64, 3)
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(64, 32, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class LargeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(3, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.pool1 = nn.AvgPool2d((2, 2))
        self.conv3 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            FeedForwardNet2NetDeepenBlock(32, 10), FeedForwardNet2NetDeepenBlock(10, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
