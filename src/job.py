#!/usr/bin/env python3.8

import models
import data
import torch.optim as optim


class Job:
    def __init__(self, config):
        config = config["job"]
        self.model = getattr(models, config["model"])()
        load_fn = getattr(data, config["dataset"])
        batch_size = config.get("batchsize", 64)
        self.trainloader = load_fn(train=True, batch_size=batch_size)
        self.testloader = load_fn(train=False, batch_size=batch_size)
        self.optimizer = getattr(optim, config.get("optimizer", "Adam"))
        self.optimizer_args = config.get("optimizer_args", dict())
        self.learning_rate_decay = config.get("learning_rate_decay", 0.9)

if __name__ == "__main__":
    Job({"job": {
        "model": "ConvNet",
        "dataset": "cifar10",
    }})