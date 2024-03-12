#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -e log/genetic_words.out-%A
#SBATCH -o log/genetic_words.out-%A

eval $(spack load --sh miniconda3)
source activate language

output=Gen_${SLURM_JOB_ID}
mkdir -p $output

datafile=Data/activity.csv
motiffile=Data/Motifs/summarized_all_motifs.parquet


python3 src/genetic_words.py $datafile $motiffile $output
