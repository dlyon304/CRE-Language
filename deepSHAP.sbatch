#!/bin/bash
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH -o log/shap.out-%A
#SBATCH -e log/shap.out-%A
#SBATCH -J SHAPPY

# LOAD SPACK ENV
eval $(spack env activate --sh tensorflow-gpu)

output_dir=modisco_access
datafile=Data/ATAC/ATAC.csv
folder=Runs/atac_300

mkdir -p $output_dir
python3 deepSHAP.py $output_dir $datafile $folder --LABEL_KEY open
# cd $output_dir
# modisco motifs -s ohe.npy -a sv.npy -n 2000 -w 100 -o modisco_results.h5
# modisco report -i modisco_results.h5 -o report/ -s report/
