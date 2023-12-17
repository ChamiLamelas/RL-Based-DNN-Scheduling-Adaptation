#!/usr/bin/env python3.8

import embedding
import models
from pathlib import Path
import torch.nn as nn


def main():
    Path(embedding.VOCABS).mkdir(exist_ok=True, parents=True)
    vocab = embedding.Vocabulary(
        "models",
        [
            models.ConvNet(),
            models.LargeConvNet(),
            models.FeedForwardNet(),
            models.FeedForwardNet2(),
            models.FeedForwardNet3(),
        ],
    )
    embedder = nn.Embedding(vocab.size(), 16)
    vocab.save(embedder.state_dict())


if __name__ == "__main__":
    main()
