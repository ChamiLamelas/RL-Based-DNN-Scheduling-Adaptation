#!/usr/bin/env python3.8

import training
import environment
import agent
import toml
import argparse
import os
from logger import ML_Logger
import shutil
from pathlib import Path
import heuristics
import seed
import torch
import copy

RESULTS = os.path.join("..", "results")


def file(f):
    if not os.path.isfile(f):
        raise argparse.ArgumentTypeError(f"{f} is not a file")
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", type=file, help="configuration file")
    return parser.parse_args()


def setup_folder(config, configfile):
    test_dir = config["folder"].startswith("test")
    config["folder"] = os.path.join(RESULTS, config["folder"])
    if os.path.isdir(config["folder"]):
        if test_dir:
            print("Specified a test folder -- deleting!")
            shutil.rmtree(config["folder"])
        else:
            raise RuntimeError(
                f"{config['folder']} already exists -- please delete it or specify a different folder"
            )
    logger = ML_Logger(log_folder=config["folder"], persist=False)
    Path(config["folder"]).mkdir(exist_ok=True, parents=True)
    if "desc" in config:
        Path(os.path.join(config["folder"], "description.txt")).write_text(
            config["desc"]
        )
    shutil.copyfile(configfile, os.path.join(config["folder"], "config.toml"))
    return logger


def loadconfig(args):
    return toml.load(args.configfile)


def get_agent(config, run_config):
    for module in [agent, heuristics]:
        if hasattr(module, run_config["agent"]):
            return getattr(module, run_config["agent"])(config)
    raise RuntimeError("could not find agent")


def run(config, run_config, logger):
    total_eps = run_config["total_episodes"]
    max_digits = len(str(total_eps))
    env = environment.Environment(config)
    agt = get_agent(config, run_config)
    agt_logger = copy.deepcopy(logger)
    agt_logger.start("agent", "agent", "learning")
    for ep in range(total_eps):
        agt.init()
        tr = training.Trainer(config, env.scheduler, agt, logger)
        tr.train(f"training{str(ep).zfill(max_digits)}")
        obj = agt.update()
        agt_logger.info(f"episode {ep + 1}/{total_eps} finished, objective: {obj}")
        agt_logger.log_metrics({"objective": obj}, "episode", agt.policy)
    if run_config.get("save", False):
        agt.save()
    agt_logger.stop()


def main():
    # torch.autograd.set_detect_anomaly(True)
    args = get_args()
    config = loadconfig(args)
    runner_config = config["runner"]
    seed.set_seed(runner_config["seed"])
    logger = setup_folder(runner_config, args.configfile)
    run(config, runner_config, logger)


if __name__ == "__main__":
    main()
