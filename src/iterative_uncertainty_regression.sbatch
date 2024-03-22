#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH -o log/iter_reg_cnn.out-%A_%a
#SBATCH -e log/iter_reg_cnn.out-%A_%a
#SBATCH --array=1-10%3
#SBATCH -J iter_reg

# LOAD SPACK ENV
eval $(spack env activate --sh tensorflow-gpu)

if [ -z ${SLURM_ARRAY_TASK_ID} ] ; then
    fold=1
    runname=${SLURM_JOB_ID}
else
    fold=${SLURM_ARRAY_TASK_ID}
    runname=${SLURM_ARRAY_JOB_ID}
fi

iterations=18
folder=Runs/$1
test_runs=100

# if [ $fold -eq 1 ]
# then
#     mkdir -p $folder
#     cp src/iterative_uncertainty_regression.py $folder/
# fi

datafile=Data/activity.parquet

for mode in Uncertainty Random
do
    i=0
    dirname=$folder/$mode/$fold
    mkdir -p $dirname
    while [ True ]
    do
        python3 src/iterative_uncertainty_regression.py $dirname $datafile $mode $i --fold $fold --FEATURE_KEY sequence --LABEL_KEY expression_log2 --num_test $test_runs --sampling_size 4000

        if [ $i -eq $iterations ]
        then
            break
        fi
        ((i++))
    done
done


