#!/usr/bin/env python3.8


import time


class Scheduler:
    def __init__(self, config):
        config = config["scheduler"]
        self.running_time = config["running_time"]
        self.gpu_changes = config.get("gpu_changes", list())
        self.start_offset = config.get("start_time", 0)
        self.start_time = None
        self.change = None

    def time_budget(self):
        return self.running_time

    def start(self):
        self.start_time = time.time() - self.start_offset
        self.change = 0

    def checkstart(self):
        assert self.start_time is not None, "run start( ) first"

    def time_left(self):
        self.checkstart()
        self.curr_time = time.time() - self.start_time
        return max(self.running_time - self.curr_time, 0)

    def allocation(self):
        self.checkstart()
        self.curr_time = time.time() - self.start_time
        output = "same"
        if (
            self.change < len(self.gpu_changes)
            and self.curr_time >= self.gpu_changes[self.change]["time"]
        ):
            output = self.gpu_changes[self.change]["change"]
            self.change += 1
        return output


if __name__ == "__main__":
    s = Scheduler(
        {
            "scheduler": {
                "running_time": 10,
                "gpu_changes": [{"time": 5, "change": "up"}],
            }
        }
    )

    s.start()
    for i in range(10):
        time.sleep(1)
        print(f"{s.allocation()} {s.time_left()}")
