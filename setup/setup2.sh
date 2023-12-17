#!/usr/bin/env bash

# Does first round of dependency setup

# OS updates inc NVIDIA drivers
sudo apt-get update
sudo apt -y install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
#sudo apt -y install nvidia-cuda-toolkit

sudo reboot

