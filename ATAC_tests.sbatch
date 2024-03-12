#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -e log/ATAC_test.err-%A
#SBATCH -o log/ATAC_test.out-%A

eval $(spack env activate --sh tensorflow)

python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive_StrongEnhancer-WeakEnhancer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive_StrongEnhancer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive_WeakEnhancer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive_Silencer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive_StrongEnhancer-WeakEnhancer-Silencer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Inactive-WeakEnhancer_StrongEnhancer-Silencer
python3 src/tf_tools/ATAC_model_eval.py ATAC7_164_10663735 Data/activity.csv ATAC7_164_10663735/5/cnn_model.keras Silencer_StrongEnhancer-WeakEnhancer




