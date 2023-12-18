#!/usr/bin/env python3.8

import pickle
import torch.nn as nn
import torch
import tracing
import models


obj = pickle.load(open("../vocabs/models.ids.pkl", "rb"))

print(obj)


layers = tracing.get_all_layers(models.ConvNet3())
layers.append(nn.Conv2d(64, 64, 3, padding=(1, 1), stride=(1, 1)))

newlayers = list(filter(lambda e: e not in obj, [str(l) for l in layers]))

new_obj = obj.copy()
new_obj.update({l: i for i, l in enumerate(newlayers, start=len(layers))})

print(new_obj)


embedder = nn.Embedding(len(obj), 16)
embedder.weight.requires_grad = False
embedder.load_state_dict(torch.load(open("../vocabs/models.embeddings.pkl", "rb")))

print(embedder.weight)


embedder2 = nn.Embedding(len(new_obj), 16)
embedder2.weight.requires_grad = False

print(embedder2.weight)


embedder2.weight[: len(obj), :] = embedder.weight

print(embedder2.weight)

pickle.dump(new_obj, open("../vocabs/models.ids.pkl", "wb+"))
torch.save(embedder2.state_dict(), "../vocabs/models.embeddings.pkl")
