#!/usr/bin/env python3.8

import math


def acc_to_reward(acc):
    # DEBUG! 
    return math.tan(acc * (math.pi / 2)) * 100
    # return acc 


if __name__ == "__main__":
    print(acc_to_reward(0.9))