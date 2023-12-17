#!/usr/bin/env bash 

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"
cd ../src
./simulationlearning.py ../config/simulation/convnet2/early_learn.toml
./simulationlearning.py ../config/simulation/convnet2/middle_learn.toml
./simulationlearning.py ../config/simulation/convnet2/late_learn.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_late_learn1.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_late_learn10.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn1.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn10.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn100.toml
./simulationlearning.py ../config/simulation/convnet2/early_to_middle_learn1000.toml
./simulationlearning.py ../config/simulation/convnet2/late_to_early_learn.toml
./simulationlearning.py ../config/simulation/convnet2/late_to_middle_learn.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn1.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn10.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn100.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_early_learn1000.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn1.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn10.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn100.toml
./simulationlearning.py ../config/simulation/convnet2/middle_to_late_learn1000.toml
printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"