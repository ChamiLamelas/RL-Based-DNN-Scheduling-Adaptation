#!/usr/bin/env python3.8

import subprocess
import torch 
import sys 

output = subprocess.check_output(["nvidia-smi"])
print("nvidia-smi")
print(output.decode("utf-8"))

print("torch version")
print(torch.__version__)

print("python version")
print(sys.version)

print("cuda device count")
print(torch.cuda.device_count())
