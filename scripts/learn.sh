#!/usr/bin/env bash 

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"

cd ../src

# Regular learning (no transfer)
# ./simulationlearning.py ../config/simulation/convnet2/early_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/late_learn.toml

# E->L learning 
# ./simulationlearning.py ../config/simulation/convnet2/early_to_late_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_late_learn10.toml

# E->M learning
# ./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn10.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn100.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn1000.toml

# L->E, L->M learning
# ./simulationlearning.py ../config/simulation/convnet2/late_to_early_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/late_to_middle_learn.toml

# M->E learning
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn10.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn100.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn1000.toml

# M->L learning
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn10.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn100.toml
# ./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn1000.toml

# L->M->E, E->L->M, E->M->L, L->E->M learning 
# ./simulationlearning.py ../config/simulation/convnet2/late_to_middle_to_early_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_late_to_middle_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_middle_to_late_learn.toml
# ./simulationlearning.py ../config/simulation/convnet2/late_to_early_to_middle_learn.toml

# L->M->E, E->L->M with diff Ts
# ./simulationlearning.py ../config/simulation/convnet2/late_to_middle_to_early_learn10.toml
# ./simulationlearning.py ../config/simulation/convnet2/early_to_late_to_middle_learn10.toml

# E->L->M transfer to convnet3
# ./simulationlearning.py ../config/simulation/convnet3/early_learn2.toml
# ./simulationlearning.py ../config/simulation/convnet3/middle_learn2.toml
# ./simulationlearning.py ../config/simulation/convnet3/late_learn2.toml
./simulationlearning.py ../config/simulation/convnet3/early_learn2_100.toml
./simulationlearning.py ../config/simulation/convnet3/middle_learn2_100.toml
./simulationlearning.py ../config/simulation/convnet3/late_learn2_100.toml

# L->M->E transfer to convnet3
# ./simulationlearning.py ../config/simulation/convnet3/early_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet3/middle_learn1.toml
# ./simulationlearning.py ../config/simulation/convnet3/late_learn1.toml
./simulationlearning.py ../config/simulation/convnet3/early_learn1_100.toml
./simulationlearning.py ../config/simulation/convnet3/middle_learn1_100.toml
./simulationlearning.py ../config/simulation/convnet3/late_learn1_100.toml

printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"