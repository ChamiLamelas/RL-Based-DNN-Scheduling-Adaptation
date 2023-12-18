#!/usr/bin/env python3.8

import torch.nn as nn
import models
import gpu
import embedding
import torch


class NetworkEncoder(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size):
        super().__init__()
        self.vocab = vocab
        self.embedder = nn.Embedding(vocab.size(), embedding_size)
        self.embedder.weight.requires_grad = False
        if vocab.has_embeddings():
            self.embedder.load_state_dict(vocab.get_embeddings())
        self.encoder = nn.LSTM(
            embedding_size, hidden_size, bidirectional=True, batch_first=True
        )

    def forward(self, layers):
        embeddings = self.embedder(layers)
        return self.encoder(embeddings)[0]


if __name__ == "__main__":
    vocab = embedding.Vocabulary("models")
    encoder = NetworkEncoder(vocab, 16, 2).to(gpu.get_device())
    print("2:")
    print("embeddings:")
    print(encoder.embedder(vocab.id(models.ConvNet2()).to(gpu.get_device())))
    print("encodings")
    print(encoder(vocab.id(models.ConvNet2()).to(gpu.get_device())))

    print("3:")
    print("embeddings:")
    print(encoder.embedder(vocab.id(models.ConvNet3()).to(gpu.get_device())))
    print("encodings")
    print(encoder(vocab.id(models.ConvNet3()).to(gpu.get_device())))

    # vocab = embedding.Vocabulary("models2")
    # encoder = NetworkEncoder(vocab, 16, 2).to(gpu.get_device())
    # print("2:")
    # print("embeddings:")
    # print(encoder.embedder(vocab.id(models.ConvNet2()).to(gpu.get_device())))
    # print("encodings")
    # print(encoder(vocab.id(models.ConvNet2()).to(gpu.get_device())))
    # print("3:")
    # print("embeddings:")
    # print(encoder.embedder(vocab.id(models.ConvNet3()).to(gpu.get_device())))
    # print("encodings")
    # print(encoder(vocab.id(models.ConvNet3()).to(gpu.get_device())))
