# Reinforcement Learning Based Deep Neural Network Adaptation for Machine Learning Schedulers 

This file is written in markdown and is best viewed on GitHub here: https://github.com/ChamiLamelas/RL-Based-DNN-Scheduling-Adaptation/blob/main/README.md.

## !! IMPORTANT !!

If you only have access to the code, before trying to reproduce any results below 
(even if you're not on CloudLab), please make sure you have cloned this repository in its entirety 
(not just the code) with:

```bash
git clone git@github.com:ChamiLamelas/RL-Based-DNN-Scheduling-Adaptation.git
```

The entire repository is too large to submit on Canvas.

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
```

Note, it may be safe to only do this on Ubuntu 18.04 as that is the only platform we have tested on.

## Prepare Initially Trained Models for Simulations

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `runner.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To retrain our models, run (from `scripts/`):

```bash
bash retrain.sh
```

This takes around 30 minutes, so it's recommended you do this in a `screen`.

## Run Simulations

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `runner.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To regenerate our simulation data, run (from `scripts/`): 

```bash
bash simulate.sh
```

This takes around 48 hours, so it's recommended you do this in a `screen`.

## Run Learning 

Note, this should only be done on CloudLab.

These runs will fail as we provide the simulation data files in this repository and `simulationlearning.py` will flag an existing folder. Follow its error handling and delete the specified folder if you wish to reproduce.

To rerun all the learning experiments discussed in the report, run (from `scripts/`): 

```bash
bash learn.sh
```

These should also be run in a `screen` and should take less than 3 hours.

## Reproducing Figures 

Note, this could be run off CloudLab (as it uses existing files primarily) but it has not been tested off CloudLab.

To obtain the accuracies and rankings discussed in the report, you will need to run (from `scripts/`): 

```bash
bash evaluations.sh
```

These run quickly and do not need to be run in a `screen`. You can find the rankings and accuracies in the `.stdout` files in `results/`. 

We provide a table of which `.stdout` files in `results/` are used for each figure in our report.

The table in figure 8a is manually constructed by getting the ranking information from these files:

| Row | Filename(s) | 
| --- | --- | 
| 1 | `E_L_learn2_1_on_E_L_learn2_1.stdout`, `E_L_learn2_10_on_E_L_learn2_10.stdout` |
| 2 | `E_M_learn2_1_on_E_M_learn2_1.stdout`, `E_M_learn2_10_on_E_M_learn2_10.stdout`, `E_M_learn2_100_on_E_M_learn2_100.stdout`, `E_M_learn2_1000_on_E_M_learn2_1000.stdout` |
| 3 | `M_E_learn2_1_on_M_E_learn2_1.stdout`, `M_E_learn2_10_on_M_E_learn2_10.stdout`, `M_E_learn2_100_on_M_E_learn2_100.stdout`, `M_E_learn2_1000_on_M_E_learn2_1000.stdout` |
| 4 | `M_L_learn2_1_on_M_L_learn2_1.stdout`, `M_L_learn2_10_on_M_L_learn2_10.stdout`, `M_L_learn2_100_on_M_L_learn2_100.stdout`, `M_L_learn2_1000_on_M_L_learn2_1000.stdout` |
| 5 | `L_E_learn2_1_on_L_E_learn2_1.stdout` |
| 6 | `L_M_learn2_1_on_L_M_learn2_1.stdout` |

The table in figure 12a is manually constructed by getting the ranking information from these files:

| Row | Filename | 
| --- | --- | 
| 1 | `E_learn2_on_E_learn2.stdout` |
| 2 | `M_learn2_on_M_learn2.stdout` |
| 3 | `L_learn2_on_L_learn2.stdout` |

The table in figure 12b is manually constructed by getting the ranking information from these files:

| Row | Filename | 
| --- | --- | 
| 1 | `E_learn3_on_E_learn3.stdout` |
| 2 | `M_learn3_on_M_learn3.stdout` |
| 3 | `L_learn3_on_L_learn3.stdout` |

To obtain all plots, run (from `src/`): 

```bash
./plotting.py
```

This does not take too long and does not need a `screen`.

We provide a table of which files in `plots/` correspond to the figures in our report.

| Figure | Filename |
| --- | --- |
| Figures 1-6 | `N/A` - constructed manually |
| Figure 7a | `objective_simulation_early_learn2.png` |
| Figure 7b | `objective_simulation_middle_learn2.png` |
| Figure 7c | `objective_simulation_late_learn2.png` |
| Figure 8a | see above table information |
| Figure 8b | `objective_early_to_late_transfer.png` |
| Figure 9a | `objective_early_to_late_explorations.png` |
| Figure 9b | `accuracy_early_to_late_explorations.png` |
| Figure 10a | `acc_elm_all_convnet2.png` |
| Figure 10b | `acc_lme_all_convnet2.png` |
| Figure 11a | `acc_elm_all_convnet3.png` |
| Figure 11b | `acc_lme_all_convnet3.png` |
| Figure 12 | see above table information |

## Repository Contents 

* `config/`: Configuration files for experiments, used in our scripts.
* `plots/`: Plots generated by plotting.py.
* `results/`: Holds results of evaluations, logs, accuracies, objectives, models and more.. everything done with our simulations and learning.
* `scripts/`: Holds some bash scripts that are useful for replicating results.
* `setup/`: Setup scripts if you're using CloudLab.
* `src/`: Source code for our implementation (discussed more below).

### Implementation Files

* `agent.py`: Here is where we define our agent. It contains the policy network as well as the logging and learning utilities for training our policy with REINFORCE. It also provides an abtract (base) agent that specifies an interface for interacting with our trainer. 
* `data.py`: Utilities for loading datasets.
* `decider.py`: Defines the action decider component of our policy network.
* `deepening.py`: Defines Net2Net deepening operations.
* `distillation.py`: Defines weight distillation operation (not used in our work, mentioned in future work).
* `embedding.py`: Defines our model/layer vocabulary system. 
* `encoding.py`: Defines our layer encoder network.
* `environment.py`: Defines our environment which is just a 2-tuple of a job and a scheduler.
* `file_renaming.py`: Utility script having to do with how we manage our logs and model parameters.
* `fix_vocab.py`: Another utility for setting up vocabularies.
* `gpu.py`: Utilities for interacting with GPUs.
* `heuristics.py`: Defines our simulation "agent" that can take a deterministic action sequence. 
* `job.py`: Defines a job (model, dataset, time budget).
* `logger.py`: Utilities for logging, model saving, etc. 
* `make_vocab.py`: Script for setting up our vocabulary. 
* `models.py`: Defines the models we use. 
* `plotting.py`: Script for plotting.
* `policy.py`: Defines our policy network.
* `prediction.py`: Utilities for making model predictions. 
* `reward.py`: Defines our reward function.
* `runner.py`: Script that can be used for default training as well as agent-consulted training if we were to not use simulations.
* `scheduler.py`: Defines our scheduler.
* `seed.py`: Utilities for setting the random seed.
* `simulation.py`: Defines our simulation class (manages the results of a simulation).
* `simulationevaluation.py`: Evaluates an agent based on simulation results.
* `simulationlearning.py`: Has an agent learn using simulation (versus what runner does).
* `tf_and_torch.py`: Utilities for converting between TensorFlow and PyTorch data and weight representations.
* `timeencoding.py`: Defines our time encoding function.
* `tracing.py`: Defines our model architecture/layer tracing utilities. 
* `training.py`: Defines our (trainer) training infrastructure.

Do not run either `make_vocab.py` or `fix_vocab.py`, the vocab set up is a bit brittle and if you rerun these it could mess up evaluation and learning. 

## Contact

If you have issues, please contact me (Chami Lamelas) at [Swaminathan.Lamelas@tufts.edu](mailto:Swaminathan.Lamelas@tufts.edu).
