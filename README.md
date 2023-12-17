# Reinforcement Learning Based Deep Neural Network Adaptation for Machine Learning Schedulers 

## Prerequisites 

### Generating Results 

To rerun our simulations and agent learning, you need to obtain access to a [CloudLab](https://www.cloudlab.us/) [Clemson r7525 node](http://docs.cloudlab.us/hardware.html#%28part._cloudlab-clemson%29). This is because our run configuration files have been set up so that adaptation times are set corresponding to the length of an epoch on the r7525 GV100L GPU. 

### Generating Plots 

To generate plots, you should be able to run off CloudLab as it does not require long training times or the specific time configurations in our experimental setups. However, it still requires a Linux machine with a CUDA GPU. Preferably, Ubuntu 18.04 (as that's what we tested on). Note we did not test this off CloudLab, so it is best to use CloudLab to reproduce any results as we give detailed setup instructions for there.

## Platform Setup 

### CloudLab 

All command are assumed to be run in `setup/`.

First, run:

```bash
bash setup1.sh
```

Wait for the node to reboot. Then, run:

```bash
bash setup2.sh
```

Wait for the node to reboot. Then, run: 

```bash
bash setup3.sh
```

Wait for the node to reboot. Then, run and follow all prompts: 

```bash 
bash setup4.sh
```

To check that you environment is set up properly, 

```
chmod +x testsetup.py
./testsetup.py
```

You should see:

* GPU utilization information for all available Nvidia GPUs.
* PyTorch version.
* Python version.
* CUDA GPU count.

Go into `src/` and run: 

```bash
chmod +x *.py
./make_vocab.py
```

### Non-CloudLab 

You will need to install Python 3.8 and the following packages from pip: 

* `torch`, `torchvision`, and `torchaudio` from `index-url` https://download.pytorch.org/whl/cu118
* `toml`, `matplotlib`, `tqdm`, and `pytz` as normal 

Go into `src/` and run: 

```bash
chmod +x *.py
./make_vocab.py
```

Note, it may be safe to only do this on Ubuntu 18.04 as that is the only platform we have tested on.

## Prepare Initially Trained Models for Simulations

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `runner.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To retrain our models, run (from `src/`):

```bash
./runner.py ../config/small2/early.toml
./runner.py ../config/small2/early_middle.toml
./runner.py ../config/small2/middle.toml
./runner.py ../config/small2/middle_late.toml
./runner.py ../config/small2/late.toml
./runner.py ../config/small3/early.toml
./runner.py ../config/small3/early_middle.toml
./runner.py ../config/small3/middle.toml
./runner.py ../config/small3/middle_late.toml
./runner.py ../config/small3/late.toml
```

This takes around 30 minutes, so it's recommended you do this in a `screen`.

Then run: 

```bash 
./file_renaming.py small_early2 small_early3 small_early_middle2 small_early_middle3 small_middle2 small_middle3 small_middle_late2 small_middle_late3 small_late2 small_late3
```

## Run Simulations

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `runner.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To regenerate our simulation data, run (from `src/`): 

```bash
./runner.py ../config/simulation/convnet2/early.toml
./runner.py ../config/simulation/convnet2/early_middle.toml
./runner.py ../config/simulation/convnet2/middle.toml
./runner.py ../config/simulation/convnet2/middle_late.toml
./runner.py ../config/simulation/convnet2/late.toml
./runner.py ../config/simulation/convnet3/early.toml
./runner.py ../config/simulation/convnet3/early_middle.toml
./runner.py ../config/simulation/convnet3/middle.toml
./runner.py ../config/simulation/convnet3/middle_late.toml
./runner.py ../config/simulation/convnet3/late.toml
```

This takes around 48 hours, so it's recommended you do this in a `screen`.

## Run Learning 

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `simulationlearning.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To rerun the learning exhibited in figures NEEDSWORK run (from `src/`): 

```bash
./simulationlearning.py ../config/simulation/convnet2/early_learn.toml
./simulationlearning.py ../config/simulation/convnet2/middle_learn.toml
./simulationlearning.py ../config/simulation/convnet2/late_learn.toml
```

These should also be run in a `screen` and should not take more than 20 minutes.

## Reproducing Figures and Tables 

Note, this could be run off CloudLab (as it uses existing files primarily) but it has not been tested off CloudLab.

To obtain the accuracies and rankings displayed in table NEEDSWORK, you will need to run 3 commands (from `src/`): 

```bash
./simulationevaluation.py simulation_early_learn2
```

This gives you the rankings and accuracies for the early stage learning.

```bash
./simulationevaluation.py simulation_middle_learn2
```

This gives you the rankings and accuracies for the middle stage learning.

```bash
./simulationevaluation.py simulation_late_learn2
```

This gives you the rankings and accuracies for the late stage learning.

These run quickly and should not be run in a `screen` (so you can scroll).

To obtain all plots, run (from `src/`): 

```bash
./plotting.py
```

This does not take too long and does not need a `screen`.

We provide a table of which files in `plots/` correspond to the figures in our report.

| Filename | Figure |
| --- | --- |
| NEEDSWORK | NEEDSWORK |

## Contact

If you have issues, please contact me (Chami Lamelas) at [Swaminathan.Lamelas@tufts.edu](mailto:Swaminathan.Lamelas@tufts.edu).
