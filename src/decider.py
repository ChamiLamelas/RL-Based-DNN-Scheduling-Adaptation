#!/usr/bin/env python3.8

import torch
import torch.nn as nn
import numpy as np
import tracing
import models
import sys


def make_decider_matrix(model):
    layers = tracing.get_all_layers(model)
    important_idxs = np.argwhere(list(map(tracing.is_important, layers))).flatten()
    decision_matrix = torch.zeros((len(important_idxs) + 1, len(layers) + 1))
    for i, idx in enumerate(important_idxs):
        decision_matrix[i, idx] = 1
    decision_matrix[-1, -1] = 1
    return decision_matrix


class SigmoidClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoding):
        return self.sigmoid(self.linear(encoding))


class Decider(nn.Module):
    def __init__(self, in_features, decider_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features, decider_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(decider_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, encodings):
        encodings = self.linear1(encodings)
        encodings = self.relu(encodings)
        encodings = self.linear2(encodings)
        encodings = self.softmax(encodings)
        encodings = encodings.flatten()
        return encodings


class Decider2(nn.Module):
    def __init__(
        self,
        in_features,
        decider_lstm_size,
        decider_linear_size,
        time_encoding_size,
        max_actions,
    ):
        super().__init__()
        self.lstm = nn.LSTM(in_features, decider_lstm_size)
        self.linear1 = nn.Linear(
            decider_lstm_size + time_encoding_size, decider_linear_size
        )
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(decider_linear_size, decider_linear_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(decider_linear_size, max_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        features = input["encodings"]
        features = self.lstm(features)[0][-1]
        features = torch.cat([input["other"], features]).unsqueeze(0)
        features = self.linear1(features)
        features = self.relu1(features)
        features = self.linear2(features)
        features = self.relu2(features)
        features = self.linear3(features)
        features = self.softmax(features)
        return features


if __name__ == "__main__":
    input = torch.randn((2, 3))
    print(input)

    classifier = SigmoidClassifier(3)
    print(classifier(input[0]))
    print(classifier(input))

    dm = make_decider_matrix(models.ConvNet())
    print(dm)

    input = torch.rand((4, 5))

    decider = Decider(5, 16)
    print(decider(input), decider(input).sum(), sep="\n")

    # decider2 = Decider2(5, 16, 10)
    # print(decider2(input))
