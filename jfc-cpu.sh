#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH -o jupyterFromCluster-cpu-%A.txt
#SBATCH -e jupyterFromCluster-cpu-%A.txt
#SBATCH -J jn-c-TF

# LOAD SPACK ENV
eval $(spack env activate --sh tensorflow-gpu)

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
python3 -m jupyter lab --no-browser --port=${port} --ip=${host}
