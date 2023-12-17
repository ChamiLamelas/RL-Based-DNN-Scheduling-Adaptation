#!/usr/bin/env bash 

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"
cd ../src
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
printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"