#!/bin/bash

# Code to sync the remote code with local code.
# Files to sync: 
# src/r0877229.py
# src/Reporter.py
# src/run_optuna.sh

# ssh -J r0877229@st.cs.kuleuven.be r0877229@ninove.student.cs.kuleuven.be

# Connect to ninove
echo "Copying genetic algorithm python files to remote ..."
scp -J r0877229@st.cs.kuleuven.be ./src/r0877229.py r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/src/
scp -J r0877229@st.cs.kuleuven.be ./src/Reporter.py r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/src/

echo "Copying data-files to remote"
scp -J r0877229@st.cs.kuleuven.be -r ./src/data/ r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/

echo "Copying run_optuna.sh to remote"
scp -J r0877229@st.cs.kuleuven.be -r ./remote/run_optuna.sh r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/

echo "Copying all files starting with optuna_ to remote"
scp -J r0877229@st.cs.kuleuven.be -r ./src/optuna_* r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/src/

