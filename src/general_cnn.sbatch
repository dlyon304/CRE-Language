#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH -o log/General_cnn.out-%A_%a
#SBATCH -e log/General_cnn.out-%A_%a
#SBATCH --array=1-7%3
#SBATCH -J ATACe-3

# LOAD SPACK ENV
eval $(spack env activate --sh tensorflow)

if [ -z ${SLURM_ARRAY_TASK_ID} ] ; then
    fold=1
    runname=${SLURM_JOB_ID}
else
    fold=${SLURM_ARRAY_TASK_ID}
    runname=${SLURM_ARRAY_JOB_ID}
fi

dirname=$1_${runname}/${fold}
mkdir -p $dirname

datafile=Data/ATAC/labeled_data.csv

python3 src/general_cnn.py $dirname $datafile --fold ${fold} --FEATURE_KEY sequence --LABEL_KEY open --lr 0.001
