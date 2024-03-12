#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH -o jupyterFromCluster-gpu-log-%A.txt
#SBATCH -e jupyterFromCluster-gpu-log-%A.txt
#SBATCH -J jn-g-TF

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
