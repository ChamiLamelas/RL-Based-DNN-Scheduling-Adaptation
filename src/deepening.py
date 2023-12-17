#!/usr/bin/env python3.8

import gpu
import numpy as np
import torch.nn as nn
import tf_and_torch
import tracing
import copy
import models


def _fc_only_deeper_tf_numpy(weight):
    deeper_w = np.eye(weight.shape[1])
    deeper_b = np.zeros(weight.shape[1])
    return deeper_w, deeper_b


def _fc_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    new_layer_w, new_layer_b = _fc_only_deeper_tf_numpy(weight)

    new_layer = nn.Linear(1, 1).to(gpu.get_device())
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_b, new_layer, "bias")

    return new_layer


def _conv_only_deeper_tf_numpy(weight):
    deeper_w = np.zeros(
        (weight.shape[0], weight.shape[1], weight.shape[3], weight.shape[3])
    )
    assert (
        weight.shape[0] % 2 == 1 and weight.shape[1] % 2 == 1
    ), "Kernel size should be odd"
    center_h = (weight.shape[0] - 1) // 2
    center_w = (weight.shape[1] - 1) // 2
    for i in range(weight.shape[3]):
        tmp = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3]))
        tmp[center_h, center_w, i] = 1
        deeper_w[:, :, :, i] = tmp
    deeper_b = np.zeros(weight.shape[3])
    return deeper_w, deeper_b


def _conv_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    deeper_w, deeper_b = _conv_only_deeper_tf_numpy(weight)

    new_layer = nn.Conv2d(
        in_channels=layer.out_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=1,
        padding=(layer.kernel_size[0] // 2, layer.kernel_size[1] // 2),
    )

    tf_and_torch.params_tf_ndarr_to_torch(deeper_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(deeper_b, new_layer, "bias")

    return new_layer


def deepen_block(parent, name, block, add_batch_norm):
    first_layer = block.layers[0]
    new_block = copy.deepcopy(block)
    new_block.layers = nn.Sequential()
    new_block.original = False
    if isinstance(first_layer, nn.Conv2d):
        new_block.layers.append(_conv_only_deeper(first_layer))
        if add_batch_norm:
            new_block.layers.append(nn.BatchNorm2d(first_layer.out_channels))
        new_block.layers.append(nn.ReLU())
    elif isinstance(first_layer, nn.Linear):
        new_block.layers.append(_fc_only_deeper(first_layer))
        new_block.layers.append(nn.ReLU())
    else:
        raise tracing.UnsupportedLayer(str(type(first_layer)))
    setattr(parent, name, nn.Sequential(block, new_block))


def deepen_model(model, logger=None, index=None, add_batch_norm=True):
    blocks = tracing.get_all_deepen_blocks(model)
    if index == len(blocks):
        if logger is not None:
            logger.info("no layers were added")
    else:
        hierarchy, name = blocks[index]
        block = getattr(hierarchy[-1], name)
        deepen_block(hierarchy[-1], name, block, add_batch_norm)
        if logger is not None:
            logger.info(
                f"deepening layer {index}/{len(blocks)}: {models.get_str_rep(block)}"
            )


if __name__ == "__main__":
    base = models.ConvNet2()
    print(f"base ({models.count_parameters(base)} parameters):\n{base}")

    m = copy.deepcopy(base)
    deepen_model(m, index=0)
    print(f"0 ({models.count_parameters(m)} parameters):\n{m}")

    m = copy.deepcopy(base)
    deepen_model(m, index=1)
    print(f"1 ({models.count_parameters(m)} parameters):\n{m}")

    m = copy.deepcopy(base)
    deepen_model(m, index=2)
    print(f"2 ({models.count_parameters(m)} parameters):\n{m}")

    m = copy.deepcopy(base)
    deepen_model(m, index=3)
    print(f"3 ({models.count_parameters(m)} parameters):\n{m}")
