#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -e log/naive_bayes.err-%A
#SBATCH -o log/naive_bayes.out-%A

eval $(spack load --sh miniconda3)
source activate language

output=bayes_${SLURM_JOB_ID}
mkdir -p $output

datafile=Data/activity.csv
motiffile=Data/Motifs/summarized_motifs.parquet


python3 src/naive_bayes.py $datafile $motiffile $output --test_file Data/retinopathy.csv --test_motifs Data/Motifs/retinopathy_motifs.parquet
