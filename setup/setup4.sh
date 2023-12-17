#!/usr/bin/env bash

echo THIS SCRIPT REQUIRES USER INPUT -- DONT LEAVE
# note this script hangs at various points (e.g. at eta 0:00:01 for torch, after matploatlib, etc.)

# 1st part taken from here: https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/
# the default python version is 3.6

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8

# this I figured out was necessary because otherwise torch would appear to install then show up
# nowhere in the system whether you run pip show/list on any of the installed pip versions
# I also figured out that the second step requires some pip installation to exist - so that
# is done normally first 
sudo apt-get install python3-pip
python3.8 -m pip install -U pip

# this first step is taken from here: https://pytorch.org/get-started/locally/
python3.8 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3.8 -m pip install toml matplotlib tqdm pytz kaggle datasets
git config --global user.name ChamiLamelas
git config --global user.email chami.lamelas@gmail.com
chmod +x testsetup.py ../src/run.py ../tests/test*.py ../plotting/*.py
