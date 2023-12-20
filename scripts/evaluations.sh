#!/usr/bin/env bash 

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"

cd ../src

# Evaluate original 3 learnings 
./simulationevaluation.py simulation_early_learn2
./simulationevaluation.py simulation_middle_learn2
./simulationevaluation.py simulation_late_learn2

# Evaluate transfer E->L 
./simulationevaluation.py simulation_early_to_late_learn2_1
./simulationevaluation.py simulation_early_to_late_learn2_10

# Evaluate transfer E->M
./simulationevaluation.py simulation_early_to_middle_learn2_1
./simulationevaluation.py simulation_early_to_middle_learn2_10
./simulationevaluation.py simulation_early_to_middle_learn2_100
./simulationevaluation.py simulation_early_to_middle_learn2_1000

# Evaluate transfer M->E 
./simulationevaluation.py simulation_middle_to_early_learn2_1
./simulationevaluation.py simulation_middle_to_early_learn2_10
./simulationevaluation.py simulation_middle_to_early_learn2_100
./simulationevaluation.py simulation_middle_to_early_learn2_1000

# Evaluate transfer M->L
./simulationevaluation.py simulation_middle_to_late_learn2_1
./simulationevaluation.py simulation_middle_to_late_learn2_10
./simulationevaluation.py simulation_middle_to_late_learn2_100
./simulationevaluation.py simulation_middle_to_late_learn2_1000

# Evaluate transfer L->E, L->M
./simulationevaluation.py simulation_late_to_early_learn2_1
./simulationevaluation.py simulation_late_to_middle_learn2_1

# Evaluate transfer L->M->E (on convnet2 seen, different Ts)
./simulationevaluation.py simulation_late_learn2 -m simulation_late_to_middle_to_early_learn2_1_1 
./simulationevaluation.py simulation_middle_learn2 -m simulation_late_to_middle_to_early_learn2_1_1 
./simulationevaluation.py simulation_early_learn2 -m simulation_late_to_middle_to_early_learn2_1_1 
./simulationevaluation.py simulation_late_learn2 -m simulation_late_to_middle_to_early_learn2_1_10 
./simulationevaluation.py simulation_middle_learn2 -m simulation_late_to_middle_to_early_learn2_1_10 
./simulationevaluation.py simulation_early_learn2 -m simulation_late_to_middle_to_early_learn2_1_10

# Evaluate transfer E->L->M (on convnet2 seen, different Ts)
./simulationevaluation.py simulation_late_learn2 -m simulation_early_to_late_to_middle_learn2_10_1 
./simulationevaluation.py simulation_middle_learn2 -m simulation_early_to_late_to_middle_learn2_10_1 
./simulationevaluation.py simulation_early_learn2 -m simulation_early_to_late_to_middle_learn2_10_1 
./simulationevaluation.py simulation_late_learn2 -m simulation_early_to_late_to_middle_learn2_10_10
./simulationevaluation.py simulation_middle_learn2 -m simulation_early_to_late_to_middle_learn2_10_10
./simulationevaluation.py simulation_early_learn2 -m simulation_early_to_late_to_middle_learn2_10_10 

# Evaluate transfer E->M->L
./simulationevaluation.py simulation_late_learn2 -m simulation_early_to_middle_to_late_learn2_1_1 
./simulationevaluation.py simulation_middle_learn2 -m simulation_early_to_middle_to_late_learn2_1_1 
./simulationevaluation.py simulation_early_learn2 -m simulation_early_to_middle_to_late_learn2_1_1 

# Evaluate transfer L->E->M
./simulationevaluation.py simulation_late_learn2 -m simulation_late_to_early_to_middle_learn2_1_1 
./simulationevaluation.py simulation_middle_learn2 -m simulation_late_to_early_to_middle_learn2_1_1 
./simulationevaluation.py simulation_early_learn2 -m simulation_late_to_early_to_middle_learn2_1_1 

# Evaluate transfer E->L->M on convnet2 unseen
./simulationevaluation.py simulation_early_middle2 -m simulation_early_to_late_to_middle_learn2_10_1
./simulationevaluation.py simulation_middle_late2 -m simulation_early_to_late_to_middle_learn2_10_1

# Evaluate transfer L->M->E on convnet2 unseen
./simulationevaluation.py simulation_early_middle2 -m simulation_late_to_middle_to_early_learn2_1_1
./simulationevaluation.py simulation_middle_late2 -m simulation_late_to_middle_to_early_learn2_1_1

# Evaluate E->L->M on convnet3 (unseen)
./simulationevaluation.py simulation_early3_v2 -m simulation_early_to_late_to_middle_learn2_10_1
./simulationevaluation.py simulation_early_middle3_v2 -m simulation_early_to_late_to_middle_learn2_10_1
./simulationevaluation.py simulation_middle3_v2 -m simulation_early_to_late_to_middle_learn2_10_1
./simulationevaluation.py simulation_middle_late3_v2 -m simulation_early_to_late_to_middle_learn2_10_1
./simulationevaluation.py simulation_late3_v2 -m simulation_early_to_late_to_middle_learn2_10_1

# Evaluate L->M->E on convnet3 (unseen)
./simulationevaluation.py simulation_early3_v2 -m simulation_late_to_middle_to_early_learn2_1_1
./simulationevaluation.py simulation_early_middle3_v2 -m simulation_late_to_middle_to_early_learn2_1_1
./simulationevaluation.py simulation_middle3_v2 -m simulation_late_to_middle_to_early_learn2_1_1
./simulationevaluation.py simulation_middle_late3_v2 -m simulation_late_to_middle_to_early_learn2_1_1
./simulationevaluation.py simulation_late3_v2 -m simulation_late_to_middle_to_early_learn2_1_1

# Evaluate L->M->E transfer to convnet3
./simulationevaluation.py simulation_convnet2_1_to_early_learn3
./simulationevaluation.py simulation_convnet2_1_to_middle_learn3
./simulationevaluation.py simulation_convnet2_1_to_late_learn3
./simulationevaluation.py simulation_convnet2_1_100_to_early_learn3
./simulationevaluation.py simulation_convnet2_1_100_to_middle_learn3
./simulationevaluation.py simulation_convnet2_1_100_to_late_learn3

# Evaluate E->L->M transfer to convnet3
./simulationevaluation.py simulation_convnet2_2_to_early_learn3
./simulationevaluation.py simulation_convnet2_2_to_middle_learn3
./simulationevaluation.py simulation_convnet2_2_to_late_learn3
./simulationevaluation.py simulation_convnet2_2_100_to_early_learn3
./simulationevaluation.py simulation_convnet2_2_100_to_middle_learn3
./simulationevaluation.py simulation_convnet2_2_100_to_late_learn3

# Evaluation scratch convnet3
./simulationevaluation.py simulation_early_learn3
./simulationevaluation.py simulation_middle_learn3
./simulationevaluation.py simulation_late_learn3

printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"
