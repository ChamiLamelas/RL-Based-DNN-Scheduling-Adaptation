#!/usr/bin/env python3.8

import argparse
import os


def result_folder(folder):
    folder = os.path.join("..", "results", folder)
    if os.path.isdir(folder):
        return folder
    raise argparse.ArgumentTypeError(f"{folder} does not exist")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=result_folder, nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    for folder in args.folders:
        os.rename(
            os.path.join(folder, "training0.finalmodel.pt"),
            os.path.join(folder, "weights.pth"),
        )
        os.rename(
            os.path.join(folder, "training0.epoch.last_runtime"),
            os.path.join(folder, "runtimes"),
        )


if __name__ == "__main__":
    main()
