#!/usr/bin/env python3.8

import torch.nn.functional as F
import torch
from tqdm import tqdm
import prediction
import models
import gpu
import copy
import distillation
import torch.optim as optim
import job
import deepening
import time
import logger
import tracing


class Trainer:
    def __init__(self, config, scheduler, agent, logger):
        self.job = job.Job(config)
        self.logger = logger
        self.config = config.get("trainer", dict())
        self.optimizer_fn = getattr(optim, config.get("optimizer", "Adam"))
        self.optimizer_args = config.get("optimizer_args", {"lr": 0.01})
        self.learning_rate_decay = config.get("learning_rate_decay", 0.9)
        self.update_optimizer(None, False)
        self.T = config.get("T", 2)
        self.start_soft_target_loss_weight = config.get("soft_target_loss_weight", 0.75)
        self.kd_weight_decay = config.get("kd_weight_decay", 0.25)
        self.soft_target_loss_weight = None
        self.mode = "initial"
        self.scheduler = scheduler
        self.agent = agent
        self.stopped_early = False
        self.allocation = None
        self.last_runtime = None
        self.teacher = None
        self.smaller = list()
        self.epoch = 0
        self.lr_scale = config.get("lr_scale", 1)
        self.min_lr = config.get("min_lr", 1e-8)

    def adapt_up(self):
        action, probabilities = self.agent.action(
            {
                "model": self.job.model,
                "totaltime": self.scheduler.running_time,
                "timeleft": self.scheduler.time_left(),
            }
        )
        deepening.deepen_model(
            self.job.model,
            self.logger,
            action,
        )
        self.update_optimizer("up", True)
        self.logger.log_metrics({"action": action}, "epoch")
        self.logger.log_metrics({"probabilities": probabilities}, "epoch")

    def train(self, log_file):
        self.logger.start(task=log_file, log_file=log_file, metrics_file=log_file)
        if "weights" in self.config:
            self.job.model.load_state_dict(torch.load(self.config["weights"]))
            self.last_runtime = logger.get_final_metric(self.config["runtimes"])
            self.logger.info(
                f"Initialized model with weights: {self.config['weights']}"
            )
        self.scheduler.start()
        while not self.stopped_early:
            self.logger.debug(
                f"Current model size: {models.count_parameters(self.job.model)} parameters"
            )
            self.logger.debug(f"Current training mode: {self.mode}")
            if self.mode == "increased":
                self.adapt_up()
            elif self.mode == "decreased":
                self.soft_target_loss_weight *= self.kd_weight_decay
                self.logger.debug(
                    f"KD weight decreased to {self.soft_target_loss_weight:.4e}"
                )
            self.train_epoch()
            self.epoch += 1
            if self.allocation == "up":
                self.mode = "increased"
                self.smaller.append(copy.deepcopy(self.job.model))
            elif self.allocation == "down":
                teacher = copy.deepcopy(self.job.model)
                self.job.model = self.smaller[-1]
                self.update_optimizer("down", True)
                distillation.deeper_weight_transfer(teacher, self.job.model)
                self.soft_target_loss_weight = self.start_soft_target_loss_weight
                self.logger.debug(
                    f"KD weight set to {self.soft_target_loss_weight:.4e}"
                )
                self.mode = "decreased"
            if not self.stopped_early and self.allocation == "same":
                test_acc = prediction.predict(
                    self.job.model, self.job.testloader, self.agent.device
                )
                if self.mode == "increased":
                    self.agent.save_prob()
                self.logger.log_metrics({"test_acc": test_acc}, "epoch", self.job.model)
        self.agent.record_acc(test_acc)
        self.logger.stop()

    def update_optimizer(self, scale, log):
        if scale == "up":
            self.optimizer_args["lr"] /= self.lr_scale
        elif scale == "down":
            self.optimizer_args["lr"] *= self.lr_scale
        self.optimizer = self.optimizer_fn(
            self.job.model.parameters(), **self.optimizer_args
        )
        self.learning_rate = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.learning_rate_decay
        )
        if log:
            self.logger.debug(
                f"Model size was changed to {models.count_parameters(self.job.model)} parameters"
            )
            self.logger.debug(f"Learning rate is now {self.optimizer_args['lr']:.6f}")

    def train_epoch(self):
        self.job.model = self.job.model.to(self.agent.device)
        self.job.model.train()
        total_correct = 0
        total_size = 0
        ti = time.time()
        for data, target in tqdm(
            self.job.trainloader,
            desc=f"training epoch {self.epoch}",
            total=len(self.job.trainloader),
        ):
            self.allocation = self.scheduler.allocation()
            if self.scheduler.time_left() == 0 or self.allocation != "same":
                if self.allocation == "same":
                    self.stopped_early = True
                    self.logger.info("training has run out of time")
                else:
                    self.logger.info(f"GPU allocation changing : going {self.allocation}")
                break
            data, target = gpu.move(self.agent.device, data, target)
            self.optimizer.zero_grad()
            loss, correct = self.compute_loss(data, target, self.teacher)
            loss.backward()
            self.optimizer.step()
            total_correct += correct
            total_size += data.size()[0]
        if not self.stopped_early and self.allocation == "same":
            if self.learning_rate.get_last_lr()[0] > self.min_lr:
                self.learning_rate.step()
                self.logger.info(
                    f"New learning rate: {self.learning_rate.get_last_lr()[0]}"
                )
            self.last_runtime = time.time() - ti
            self.logger.log_metrics({"train_acc": total_correct / total_size}, "epoch")
            self.logger.log_metrics({"last_runtime": self.last_runtime}, "epoch")

    def compute_loss(self, data, target, teacher):
        student_logits = self.job.model(data)
        correct = prediction.num_correct(student_logits, target)
        loss = F.cross_entropy(student_logits, target)
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(data)
            soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
            soft_prob = F.log_softmax(student_logits / self.T, dim=-1)
            soft_targets_loss = (
                -torch.sum(soft_targets * soft_prob)
                / soft_prob.size()[0]
                * (self.T**2)
            )
            loss = (
                self.soft_target_loss_weight * soft_targets_loss
                + (1 - self.soft_target_loss_weight) * loss
            )
        return loss, correct
