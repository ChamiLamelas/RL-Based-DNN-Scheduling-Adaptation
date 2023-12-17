#!/usr/bin/env python3.8

import agent
import tracing
import random
import seed
import models
import torch
import itertools
from collections import defaultdict
import pickle
import simulation
import os 


class RandomAgent(agent.BaseAgent):
    def __init__(self, config):
        super().__init__(config)

    def init(self):
        pass

    def action(self, state):
        nblocks = len(tracing.get_all_deepen_blocks(state["model"]))
        a = random.randint(0, nblocks)
        p = torch.zeros(nblocks + 1)
        p[a] = 1
        return a, p.tolist()

    def record_acc(self, acc):
        pass

    def update(self):
        pass

    def save_prob(self):
        pass

    def save(self):
        pass


class DeterministicAgent(agent.BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.actions = self.config["action_sequence"]
        self.curr_action = None

    def init(self):
        self.curr_action = 0

    def action(self, state):
        assert self.curr_action is not None, "run init( ) first"
        nblocks = len(tracing.get_all_deepen_blocks(state["model"]))
        if self.curr_action >= len(self.actions):
            a = nblocks
        else:
            a = self.actions[self.curr_action]
            self.curr_action += 1
        p = torch.zeros(nblocks + 1)
        p[a] = 1
        return a, p.tolist()

    def record_acc(self, acc):
        pass

    def update(self):
        pass

    def save_prob(self):
        pass

    def save(self):
        pass


class SimulationAgent(agent.BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.action_set_size = self.config["action_set_size"]
        self.num_actions = self.config["num_actions"]
        self.save_file = self.config["save_file"]
        if os.path.isfile(self.save_file):
            raise RuntimeError(f"{self.save_file} exists!")
        actions = list(range(self.action_set_size))
        iterables = [actions] * self.num_actions
        self.action_sequences = list(itertools.product(*iterables))
        self.current_sequence = -1
        self.current_action_in_seq = None
        self.data = defaultdict(lambda: {"timeleft": []})

    def init(self):
        self.current_action_in_seq = 0
        self.current_sequence += 1
        assert self.current_sequence < len(self.action_sequences)

    def action(self, state):
        assert self.current_action_in_seq is not None, "run init( ) first"
        actions = self.action_sequences[self.current_sequence]
        if self.current_action_in_seq >= len(actions):
            a = self.action_set_size - 1
        else:
            a = actions[self.current_action_in_seq]
            self.current_action_in_seq += 1
            self.data[actions]["timeleft"].append(state["timeleft"])
        if "totaltime" not in self.data[actions]:
            self.data[actions]["totaltime"] = state["totaltime"]
        p = torch.zeros(self.action_set_size)
        p[a] = 1
        return a, p.tolist()

    def record_acc(self, acc):
        self.data[self.action_sequences[self.current_sequence]]["acc"] = acc

    def update(self):
        pass

    def save_prob(self):
        pass

    def save(self):
        with open(self.save_file, "wb+") as f:
            pickle.dump(dict(self.data), f)

    def get_num_supported_episodes(self):
        return len(self.action_sequences)


if __name__ == "__main__":
    seed.set_seed()
    states = [
        {"model": models.ConvNet(), "totaltime": 400, "timeleft": 50},
        {"model": models.ConvNet(), "totaltime": 400, "timeleft": 25},
    ]

    r = RandomAgent({"agent": {"device": 0}, "runner": {"folder": "test"}})
    r.init()
    print(r.action(states[0]))
    print(r.action(states[1]))

    d = DeterministicAgent(
        {"agent": {"device": 0, "action_sequence": [3]}, "runner": {"folder": "test"}}
    )
    d.init()
    print(d.action(states[0]))
    print(d.action(states[1]))

    s = SimulationAgent(
        {
            "agent": {
                "action_set_size": 3,
                "num_actions": 2,
                "save_file": "test.pkl",
                "device": 0,
            },
            "runner": {"folder": "test"},
        }
    )
    print(s.get_num_supported_episodes())
    for i in range(9):
        s.init()
        for j in range(2):
            a, p = s.action(states[j])
            print(f"episode {i} : action, probabilities {j} : {a}, {p}")
        s.record_acc(i / 10)
    s.save()

    sr = simulation.Simulation({"simulation": {"datafile": "test.pkl"}})
    print(sr.get_total_time())

    for e in [[]] + list(map(lambda x: [x], [0, 1, 2])):
        print(f"time left {e} : {sr.get_time_left(e)}")

    for i in range(3):
        for j in range(3):
            e = (i, j)
            print(f"acc {e} : {sr.get_acc(e)}")

    print("\n".join(f"{k} {v}" for (k, v) in sr.get_acc_ranking()))
