#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH -o jupyterFromCluster-log-%A.txt
#SBATCH -e jupyterFromCluster-log-%A.txt
#SBATCH -J jn-c-LNG

# LOAD ANACONDA MODULE
eval $(spack load --sh miniconda3)
unset PYTHONPATH
unset XDG_RUNTIME_DIR

# ACTIVATE CONDA ENVIRONMENT
source activate language

# CREATE PORT AND GET NAME OF SERVER NODE
port=$(shuf -i9000-9999 -n1)
host=$(hostname)

# OUTPUT SERVER AND PORT
echo node: "$host"
echo port: "$port"


# Print tunneling instructions to ~/logs/jupyterFromCluster-log-{jobid}.txt
#echo -e "
#    Run in your local terminal to create an SSH tunnel to $host
#    -----------------------------------------------------------
#    ssh -N -L $port:$host:$port $USER@login.htcf.wustl.edu
#    -----------------------------------------------------------
#
#    Go to the following address in a browser on your local machine
#    --------------------------------------------------------------
#    https://localhost:$port
#    --------------------------------------------------------------
#    "

# Launch Jupyter lab server
jupyter lab --no-browser --port=${port} --ip=${host}
