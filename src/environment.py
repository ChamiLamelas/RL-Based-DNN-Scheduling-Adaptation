#!/usr/bin/env python3.8

import scheduler 
import job 

class Environment:
    def __init__(self, config):
        self.scheduler = scheduler.Scheduler(config)
        self.job = job.Job(config)