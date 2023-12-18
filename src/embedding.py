#!/usr/bin/env python3.8

import torch.nn as nn
import torch
import sys
import os
import tracing
import models
import pickle

VOCABS = os.path.join("..", "vocabs")
IDS = ".ids.pkl"
EMBEDDINGS = ".embeddings.pkl"
NONE_LAYER_ID = 0

class Vocabulary:
    def __init__(self, name=None, models=None):
        self.ids_file = os.path.join(VOCABS, name + IDS)
        self.embeddings_file = os.path.join(VOCABS, name + EMBEDDINGS)
        self.embeddings = None
        if models is None:
            self.load_from_files()
        else:
            self.init_from_models(models)

    def load_from_files(self):
        if os.path.isfile(self.ids_file):
            with open(self.ids_file, "rb") as f:
                self.ids = pickle.load(f)
        else:
            raise RuntimeError(f"{self.ids_file} does not exist")
        if os.path.isfile(self.embeddings_file):
            with open(self.embeddings_file, "rb") as f:
                self.embeddings = torch.load(f)

    def init_from_models(self, models):
        self.ids = dict() # {"none-layer": NONE_LAYER_ID}
        for model in models:
            for layer in tracing.get_all_layers(model):
                key = str(layer)
                if key not in self.ids:
                    self.ids[key] = len(self.ids)

    def id(self, model):
        ids = []
        for layer in tracing.get_all_layers(model):
            key = str(layer)
            if key in self.ids:
                ids.append(self.ids[key])
            else:
                raise RuntimeError(f"{key} does not have an ID in the vocabulary")
        return torch.LongTensor(ids)

    def save(self, embeddings=None):
        if embeddings is not None:
            torch.save(embeddings, self.embeddings_file)
        with open(self.ids_file, "wb+") as f:
            pickle.dump(self.ids, f)

    def size(self):
        return len(self.ids)

    def has_embeddings(self):
        return self.embeddings is not None

    def get_embeddings(self):
        return self.embeddings


if __name__ == "__main__":
    # vocab = Vocabulary("test", [models.ConvNet2()])
    # print(vocab.has_embeddings())

    # embedder = nn.Embedding(vocab.size(), 4)
    # vocab.save(embedder.state_dict())

    # vocab2 = Vocabulary("test")
    # print(vocab2.has_embeddings())

    vocab = Vocabulary("models")
    