# give me the deeplift shappy scorees
# Daniel Lyon | WUSTL Cohen Lab | Mar, 2024

import os
import argparse
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
import shap



def one_hot_seqs(seqs) -> np.array:
    static_1hotmap = {
        'A' : np.array([1,0,0,0]),
        'a' : np.array([1,0,0,0]),
        'C' : np.array([0,1,0,0]),
        'c' : np.array([0,1,0,0]),
        'G' : np.array([0,0,1,0]),
        'g' : np.array([0,0,1,0]),
        'T' : np.array([0,0,0,1]),
        't' : np.array([0,0,0,1]),
    }
    onehot_seqs = []
    for seq in seqs:
        onehot_seqs.append(
            [static_1hotmap[seq[i]] if seq[i] in static_1hotmap.keys() else static_1hotmap[random.choice(['A','C','G','T'])] for i in range(len(seq))]
        )
    return np.stack(onehot_seqs)



def main(output_dir, data_file,folder,fold,FEATURE_KEY,LABEL_KEY):
    
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
    
    activity_df = pd.read_csv(data_file, index_col=0)
    n = int(len(activity_df)/2)
    
    test_df = activity_df[activity_df['test_set']]
    validation_df = activity_df[activity_df['validation_set']]
    train_df = activity_df[activity_df['train_set']]

    x = one_hot_seqs(activity_df[FEATURE_KEY])
    x_test = one_hot_seqs(test_df[FEATURE_KEY])
    x_validation = one_hot_seqs(validation_df[FEATURE_KEY])
    x_train = one_hot_seqs(train_df[FEATURE_KEY])

    y_test = test_df[LABEL_KEY].values
    y_validation = validation_df[LABEL_KEY].values
    y_train = train_df[LABEL_KEY].values
    
    model = keras.models.load_model(os.path.join(folder,fold,"cnn_model.keras"))
    
    explainer = shap.DeepExplainer(model, x_validation[:100])

    sv = explainer.shap_values(x_train[n:])

    seq_len = len(activity_df.iloc[0]['sequence'])
    sv = np.reshape(sv, (len(x_train[n:]),seq_len,4))
    
    seq_out = x_train[n:].transpose(0,2,1)
    sv_out = sv.transpose(0,2,1)
    np.save(os.path.join(output_dir,'ohe.npy'), seq_out)
    np.save(os.path.join(output_dir,'sv.npy'), sv_out)
    
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    parser.add_argument("data_file", type=str, help='Path to file with features and lables')
    parser.add_argument("folder", type=str, help='folder with the keras model')
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fold", type=str, default='1')
    parser.add_argument("--FEATURE_KEY", type=str, default='sequence', help="Column name(s) fo feature for model input")
    parser.add_argument("--LABEL_KEY", type=str, default='activity_bin')
    # parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args = parser.parse_args()
    
    main(**vars(args))










