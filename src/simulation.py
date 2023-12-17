import pickle
import toml
import os


class Simulation:
    def __init__(self, config):
        if "simulation" in config:
            file = config["simulation"]["datafile"]
        else:
            file = config["agent"]["save_file"]
        with open(file, "rb") as f:
            self.data = pickle.load(f)
        self.num_actions = len(list(self.data.keys())[0])
        self.placeholder = [0] * self.num_actions

    def get_time_left(self, actions):
        assert len(actions) < self.num_actions
        insert_len = len(actions)
        placeholder = self.placeholder.copy()
        placeholder[:insert_len] = actions
        placeholder = tuple(placeholder)
        assert (
            "timeleft" in self.data[placeholder]
        ), "action( ) wasn't called on SimulationAgent"
        time_left = self.data[placeholder]["timeleft"]
        return time_left[insert_len]

    def get_acc(self, actions):
        actions = tuple(actions)
        assert len(actions) == self.num_actions
        assert (
            "acc" in self.data[actions]
        ), "record_acc( ) wasn't called on SimulationAgent"
        return self.data[actions]["acc"]

    def get_num_actions(self):
        return self.num_actions

    def get_total_time(self):
        return self.data[tuple(self.placeholder)]["totaltime"]

    def get_acc_ranking(self):
        return sorted(
            ((k, v["acc"]) for k, v in self.data.items()),
            key=lambda e: e[1],
            reverse=True,
        )
