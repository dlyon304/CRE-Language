import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import argparse

import tensorflow as tf
import tensorboard as tb
from tensorflow import keras
from keras import datasets, layers, models, Input, Model, activations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import scipy

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


def main(output_dir, data_file, model_path, activity_splits, batch_size, FEATURE_KEY, LABEL_KEY):
    
    
    
    #Read data file and extract test set
    filename, file_extension = os.path.splitext(data_file)
    if file_extension == 'parquet':
        data_df = pd.read_parquet(data_file)
    else:
        data_df = pd.read_csv(data_file, index_col=0)
    test_df = data_df[data_df['set'] == "TEST"]
    
    # Get activity bin groups
    open_labels = activity_splits.split('_')[1].split('-')
    closed_labels = activity_splits.split('_')[0].split('-')
    test_df = test_df[test_df['activity_bin'].isin(open_labels+closed_labels)].copy()
    test_df[LABEL_KEY] = test_df['activity_bin'].apply(lambda x: "Open" if x in open_labels else "Closed")
    
    
    # Load keras model and test/evaluate
    model = keras.models.load_model(model_path)
    
    null_metrics = []
    full_batch_metrics = []
    batch_metrics = []
    data_batches = []
    rounds = ['Genomic', 'CrxMotifMutant', 'Round2', 'Round3a', 'Round3b', 'Round3c', 'Round4a', 'Round4b']
    for round in rounds:
        data_batches.append(round)
        
        #Split from activity_bin to open/close
        split_df = test_df[test_df['data_batch_name'].isin(data_batches)]
        
        
        # Prepare features and labels for testing
        x_test = one_hot_seqs(split_df[FEATURE_KEY])
        encoder = LabelEncoder()
        encoder.fit(split_df[LABEL_KEY])
        classes = encoder.classes_
        num_classes = len(classes)
        y_test = encoder.transform(split_df[LABEL_KEY])
        y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
        results = model.evaluate(x_test,y_test,batch_size=128, verbose=0)
        predictions_ = model.predict(x_test, verbose=0)
        
        truths = np.argmax(y_test, axis=1)
        predictions = np.argmax(predictions_, axis=1)
        
        full_batch_metrics.append(classification_report(truths,predictions,target_names=classes))
        report = classification_report(truths,predictions,target_names=classes, output_dict=True)
        r = report['weighted avg']
        r['accuracy'] = report['accuracy']
        batch_metrics.append(r)
        
        null_report = classification_report(truths, random.choices([0,1], k = len(truths)), target_names=classes, output_dict=True)
        nr = null_report['weighted avg']
        nr['accuracy'] = null_report['accuracy']
        null_metrics.append(nr)
        
        print("Finished round", round, "Accuracy:", r['accuracy'])
      
    # Create and save metric plot  
    for batch in [batch_metrics, null_metrics]:
        plt.rcParams['figure.figsize'] = [12, 5]
        metrics = list(batch[0].keys())
        support = [int(r['support']) for r in batch]
        metrics.remove('support')
        labels = [b+"\n"+str(sup) for b,sup in zip(rounds,support)]
        
        fig, ax = plt.subplots()
        width = 0.15
        sep = [-2,-1,0,1]
        for i in range(len(metrics)):
            
            ax.bar(
                x=np.arange(len(batch))+width*sep[i],
                height = [r[metrics[i]] for r in batch],
                width=width,
                label = metrics[i],
                align='edge',
            )
        
        plt.xticks(ticks=np.arange(len(batch)), labels=labels)
        plt.ylabel("Score")
        
        prefix = "NULL--" if batch == null_metrics else ""
        
        title = prefix + " ".join(open_labels) + ":Open | " + " ".join(closed_labels) + ":Closed"
        plt.title(title)
        plt.ylim(0,0.75)
        
        plt.legend()
        fig_dir = os.path.join(output_dir,'figures')
        os.makedirs(
            fig_dir,
            exist_ok=True)
        plt.savefig(
            os.path.join(fig_dir,prefix+activity_splits+".png")
            ,format='png')

            
    # Make combined plots
    plt.rcParams['figure.figsize'] = [12, 5]
    metrics = list(batch[0].keys())
    support = [int(r['support']) for r in batch]
    metrics.remove('support')
    labels = [b+"\n"+str(sup) for b,sup in zip(rounds,support)]
    
    fig, ax = plt.subplots()
    width = 0.15
    sep = [-2,-1,0,1]
    for i in range(len(metrics)):
        
        ax.bar(
            x=np.arange(len(batch))+width*sep[i],
            height = [mdl[metrics[i]]-nll[metrics[i]]  for mdl,nll in zip(batch_metrics, null_metrics)],
            width=width,
            label = metrics[i],
            align='edge',
        )
    
    plt.xticks(ticks=np.arange(len(batch)), labels=labels)
    plt.ylabel("Delta Score")
    
    title = "(Model-Null) " + " ".join(open_labels) + ":Open | " + " ".join(closed_labels) + ":Closed"
    plt.title(title)
    plt.ylim(-0.4,0.4)
    
    plt.legend()
    fig_dir = os.path.join(output_dir,'figures')
    os.makedirs(
        fig_dir,
        exist_ok=True)
    plt.savefig(
        os.path.join(fig_dir,"(Model-Null) "+activity_splits+".png")
        ,format='png')
    
    # Create a large log file with all metrics
    # title = " ".join(open_labels) + ":Open | " + " ".join(closed_labels) + ":Closed"    
    # reports_file = os.path.join(output_dir, "reports.txt")
    # with open(reports_file, 'a') as file:
    #     file.write(title.upper())
    #     file.write('\n')
    #     for i in range(len(rounds)):
    #         file.write(rounds[i])
    #         file.write('\n')
    #         file.write(full_batch_metrics[i])
    #         file.write('\n')
    #         file.write('\n')
    #     file.write("\n-------------------------------------------\n")
    
    

    
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    parser.add_argument("data_file", type=str, help='Path to file with features and lables')
    parser.add_argument("model_path", type=str, help="Path to Keras model")
    parser.add_argument("activity_splits", type=str, help="X-X_Y-Y")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--FEATURE_KEY", type=str, default='sequence', help="Column name(s) fo feature for model input")
    parser.add_argument("--LABEL_KEY", type=str, default='open')
    args = parser.parse_args()
    
    main(**vars(args))