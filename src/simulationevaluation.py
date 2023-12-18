#!/usr/bin/env python3.8

import argparse
import simulation
import agent
import os
import job
import toml
import torch
import deepening
import tracing
import sys


def folder(f):
    f = os.path.join("..", "results", f)
    if not os.path.isdir(f):
        raise argparse.ArgumentTypeError(f"{f} is not a folder")
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluationfolder", type=folder, help="evaluationfolder")
    parser.add_argument(
        "-m", "--modelfolder", type=folder, help="modelfolder", default=None
    )
    parser.add_argument(
        "--showstdout",
        default=False,
        type=bool,
        help="show stdout or send it to file (default to file)",
    )
    return parser.parse_args()


def get_name(evaluationfolder, modelfolder):
    def shorten(foldername):
        foldername = os.path.basename(foldername)
        foldername = foldername.replace("simulation_", "")
        foldername = foldername.replace("early_middle", "EM")
        foldername = foldername.replace("middle_late", "ML")
        foldername = foldername.replace("early", "E")
        foldername = foldername.replace("middle", "M")
        foldername = foldername.replace("late", "L")
        foldername = foldername.replace("to_", "")
        return foldername

    return os.path.join(
        "..",
        "results",
        shorten(modelfolder) + "_on_" + shorten(evaluationfolder) + ".stdout",
    )


def main():
    args = get_args()

    modelfolder = (
        args.evaluationfolder if args.modelfolder is None else args.modelfolder
    )

    config = toml.load(os.path.join(modelfolder, "config.toml"))

    old_stdout = sys.stdout
    if not args.showstdout:
        stdout_file = get_name(args.evaluationfolder, modelfolder)
        sys.stdout = open(stdout_file, "w+")

    if "policyfile" not in config["agent"]:
        config["agent"]["policyfile"] = os.path.join(modelfolder, "agent.finalmodel.pth")
    agt = agent.Agent(config)

    # agt.policy.load_state_dict(
    #     torch.load()
    # )
    agt.eval()

    eval_config = toml.load(os.path.join(args.evaluationfolder, "config.toml"))

    # model = job.Job(config).model

    # print(model)

    model = job.Job(eval_config).model

    print(model)

    sim = simulation.Simulation(eval_config)

    print(f"=== Model Folder ===\n{modelfolder}\n")
    print(f"=== Evaluation Folder ===\n{args.evaluationfolder}")

    action_set_size = len(tracing.get_all_deepen_blocks(model)) + 1

    actions = list()
    for i in range(sim.get_num_actions()):
        action, probs = agt.action(
            {
                "totaltime": sim.get_total_time(),
                "timeleft": sim.get_time_left(actions),
                "model": model,
            }
        )
        deepening.deepen_model(model, index=action)
        actions.append(action)

        print(f"=== Step {i + 1} ===")
        print(f"Action: {action}")
        print(f"Probabilities: {' '.join(f'{e:.4f}' for e in probs)}")
        print()

    actions = tuple(actions)
    no_adaptation = tuple([action_set_size - 1] * sim.get_num_actions())

    max_rank_len = len(str(len(sim.get_acc_ranking()) - 1))

    print("=== Ranking ===")
    print(
        "\n".join(
            f"rank {str(i).zfill(max_rank_len)}: {k} {v}"
            + (" ***" if no_adaptation == k or actions == k else "")
            for i, (k, v) in enumerate(sim.get_acc_ranking(), start=1)
        )
    )

    sys.stdout = old_stdout


if __name__ == "__main__":
    main()
