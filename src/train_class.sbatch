#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH -o log/class_cnn.out-%A_%a
#SBATCH -e log/class_cnn.out-%A_%a
#SBATCH --array=1-10%3
#SBATCH -J cnn_class

# LOAD SPACK ENV
eval $(spack env activate --sh tensorflow-gpu)

if [ -z ${SLURM_ARRAY_TASK_ID} ] ; then
    fold=1
    runname=${SLURM_JOB_ID}
else
    fold=${SLURM_ARRAY_TASK_ID}
    runname=${SLURM_ARRAY_JOB_ID}
fi

dirname=Runs/$1/${fold}
mkdir -p $dirname

datafile=Data/ATAC/ATAC.csv

python3 src/train_class.py $dirname $datafile --num_classes 2 --fold ${fold} --FEATURE_KEY sequence --LABEL_KEY open --lr 0.0002
