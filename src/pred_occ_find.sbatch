#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -J po_ret

eval $(spack load --sh miniconda3)
source activate language

output_file=Data/Motifs/retinopathy_all_motifs.parquet

data_file=Data/retinopathy.csv
pwm_file=Data/PWMs/All.meme


python3 src/pred_occ_find.py $data_file $pwm_file $output_file
