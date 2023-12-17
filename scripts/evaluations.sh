#!/usr/bin/env bash 

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"

cd ../src
./simulationevaluation.py simulation_early_learn2
./simulationevaluation.py simulation_middle_learn2
./simulationevaluation.py simulation_late_learn2
./simulationevaluation.py simulation_early_to_late_learn2_1
./simulationevaluation.py simulation_early_to_late_learn2_10
./simulationevaluation.py simulation_early_to_middle_learn2_1
./simulationevaluation.py simulation_early_to_middle_learn2_10
./simulationevaluation.py simulation_early_to_middle_learn2_100
./simulationevaluation.py simulation_early_to_middle_learn2_1000
./simulationevaluation.py simulation_middle_to_early_learn2_1
./simulationevaluation.py simulation_middle_to_early_learn2_10
./simulationevaluation.py simulation_middle_to_early_learn2_100
./simulationevaluation.py simulation_middle_to_early_learn2_1000
./simulationevaluation.py simulation_middle_to_late_learn2_1
./simulationevaluation.py simulation_middle_to_late_learn2_10
./simulationevaluation.py simulation_middle_to_late_learn2_100
./simulationevaluation.py simulation_middle_to_late_learn2_1000
./simulationevaluation.py simulation_late_to_early_learn2_1
./simulationevaluation.py simulation_late_to_middle_learn2_1

printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"