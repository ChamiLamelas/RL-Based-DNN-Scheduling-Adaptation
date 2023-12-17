#!/usr/bin/env python3.8

import torch.nn as nn
import models


class UnsupportedLayer(Exception):
    pass


def get_all_deepen_blocks(module):
    blocks = list()

    def _recursive_get_all_deepen_blocks(curr, hierarchy, name):
        if models.is_deepen_block(curr):
            blocks.append((hierarchy, name))
        else:
            for name, child in curr.named_children():
                _recursive_get_all_deepen_blocks(child, hierarchy + (curr,), name)

    _recursive_get_all_deepen_blocks(module, tuple(), None)
    return blocks


def is_important(module):
    return any(isinstance(module, layer_type) for layer_type in [nn.Conv2d, nn.Linear])




def get_all_important_layer_hierarchies(module):
    layers = dict()

    def _recursive_get_all_important_layer_hierarchies(hierarchy, parent, curr):
        if is_important(curr):
            layers[hierarchy] = parent
        for name, child in curr.named_children():
            _recursive_get_all_important_layer_hierarchies(
                hierarchy + (name,), curr, child
            )

    _recursive_get_all_important_layer_hierarchies(tuple(), None, module)
    return layers


def get_all_layers(module):
    layers = list()

    def _recursive_get_all_layers(curr):
        if len(list(curr.children())) == 0:
            layers.append(curr)
        for child in curr.children():
            _recursive_get_all_layers(child)

    _recursive_get_all_layers(module)
    return layers
