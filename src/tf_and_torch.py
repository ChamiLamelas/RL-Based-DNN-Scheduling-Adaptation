#!/usr/bin/env python3.8

import numpy as np
import torch.nn as nn
import torch

# look at comments of: https://stackoverflow.com/a/36223553 for TF weights shape


def params_torch_to_tf_ndarr(torch_layer, attr):
    torch_ndarr = getattr(torch_layer, attr).data.cpu().numpy()
    if attr == "bias":
        return torch_ndarr
    elif attr == "weight":
        if isinstance(torch_layer, nn.Linear):
            return np.transpose(torch_ndarr)
        elif isinstance(torch_layer, nn.Conv2d):
            return np.moveaxis(torch_ndarr, [0, 1], [3, 2])


def params_tf_ndarr_to_torch(tf_ndarr, torch_layer, attr):
    if attr == "bias":
        setattr(torch_layer, attr, nn.Parameter(torch.Tensor(tf_ndarr)))
    elif attr == "weight":
        if isinstance(torch_layer, nn.Linear):
            new_in_features, new_out_features = tf_ndarr.shape
            setattr(
                torch_layer, attr, nn.Parameter(
                    torch.Tensor(np.transpose(tf_ndarr)))
            )
            torch_layer.in_features = new_in_features
            torch_layer.out_features = new_out_features
        elif isinstance(torch_layer, nn.Conv2d):
            setattr(
                torch_layer,
                attr,
                nn.Parameter(
                    torch.Tensor(np.moveaxis(
                        tf_ndarr, [3, 2], [0, 1]))
                ),
            )
            torch_layer.in_channels = tf_ndarr.shape[2]
            torch_layer.out_channels = tf_ndarr.shape[3]
