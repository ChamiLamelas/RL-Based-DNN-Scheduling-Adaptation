#!/usr/bin/env bash

printf "start\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"

cd ../src
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
./file_renaming.py small_early2 small_early3 small_early_middle2 small_early_middle3 small_middle2 small_middle3 small_middle_late2 small_middle_late3 small_late2 small_late3

printf "end\t"
env TZ=America/New_York date +"%m/%d/%Y %I:%M:%S %p"