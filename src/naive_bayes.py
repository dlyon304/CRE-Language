# Takes in any set of sequences and labels.  Transforms them into "sentences" then creates a bayes classifier and tests performance
# Daniel Lyon | WUSTL Cohen Lab | Jan, 2024

import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.metrics import f1_score
from grammar.sentences import po_sentences
import random

SEQUENCE_COL = 'sequence'
LABEL_COL = 'activity_bin'
SPLIT_COL = 'split'
TRAIN_KEY = 'train'
TEST_KEY = 'test'


def get_counts(class_sentences: pd.api.typing.DataFrameGroupBy) -> tuple[dict[Counter], dict[float]]:
    
    class_counts = dict()
    priors = dict()
    total = 0
    
    for c, sentences in class_sentences:
        word_counter = Counter()
        [word_counter.update(s.split()) for s in sentences.to_list()]
        class_counts[c] = word_counter
        priors[c] = len(sentences)
        total += len(sentences)
        
    class_priors = dict([(c, v / total) for c,v in priors.items()])
    
    return class_counts, class_priors

def predict_classes(class_counts: dict[Counter], test_sentences: list, class_priors: pd.Series = None) -> list[str]:
    '''
    Given class word counts make log predictions on Xtest and use argmax to predict class
    ----------------------------------------------
    Inputs:
        class_counts[class] = Counter(all words in sentences)
        test_sentences = list(s1,...,sn)
        class_priors.loc[class] = float(#sentences in class / #sentences)
    Outputs:
        sentence_log_preds = list(Cpred1,...,Cpredn)
    '''
    
    classes = list(class_counts.keys())
    
    #If no priors are passed then set all class priors to zero = log(1)
    if not class_priors:
        class_priors = pd.Series(np.zeros(len(classes)), index = classes)
    
    #Determine the entire vocabulary seen or to be seen
    alphabet = set()
    [alphabet.update(class_counts[k].keys()) for k in class_counts.keys()]
    [alphabet.update(s.split()) for s in test_sentences]
    V = len(alphabet)
    
    #Convert raw counts to log probs use smoothing by adding 1 to each word in alphabet for each class
    [class_counts[c].update(alphabet) for c in classes]
    class_word_logs = dict([
        (c,
        dict([
            (word,
            math.log(count / class_counts[c].total()))
            for word,count in class_counts[c].items()
        ]))
        for c in classes
    ])
     
    # Log probs for each class using naive bayes
    sentence_log_preds = []
    for sentence in test_sentences:
        preds = class_priors.copy()
        for c in classes:
            preds[c] += sum(class_word_logs[c][word] for word in sentence.split())
        
        sentence_log_preds.append(preds.idxmax())
    
    return sentence_log_preds

def eval_bae(Ytrue: list[str], Ypred: list[str]) -> pd.Series:
    
    classes = list(set(Ytrue))
    averages = ['micro','macro','weighted']
    
    f1 = [f1_score(Ytrue, Ypred, labels=classes, average=a) for a in averages]
    
    #If there are only two classes use binary f1 as well
    if len(classes) == 2:
        f1.append(f1_score(Ytrue,Ypred,pos_label=classes[1],average='binary'))
        averages.append('binary')
    
    return pd.Series(
        data=f1,
        index=averages,
        name = 'f1_scores'
    )
    
def main(data_file, motif_file, output_dir, test_file, test_motifs):
    
    print("Read in datafile to pandas dataframe", flush=True)
    # Read in datafile to pandas dataframe.  Then convert all sequences to sentences and add on as a column
    data_df = pd.read_csv(data_file,index_col=0)
    data_df['split'] = random.choices([TRAIN_KEY, TEST_KEY], cum_weights=[0.8,1], k=len(data_df))
    data_df['sentence'] = po_sentences(data_df.index.to_list(),motif_file)

    print("Split train and test from main dataframe", flush=True)
    #Split train and test from main dataframe
    train_df = data_df[data_df[SPLIT_COL]==TRAIN_KEY]
    test_df = data_df[data_df[SPLIT_COL]==TEST_KEY]
    
    print("Write out sentences for analysis", flush=True)
    #Write out sentences for analysis
    train_df['sentence'].to_csv(os.path.join(output_dir,'train_sentences.csv'))
    test_df['sentence'].to_csv(os.path.join(output_dir,"test_sentences.csv"))
    
    print("Read training sentences to get raw counts", flush=True)
    #Read training sentences to get raw counts 
    counts, priors = get_counts(train_df.groupby(LABEL_COL)['sentence'])
    print(counts,flush=True)
    
    print("Use a naive bayesian approach to document classification with BOW encoding", flush=True)
    #Use a naive bayesian approach to document classification with BOW encoding
    test_truths = test_df[LABEL_COL].to_list()
    test_sentences = test_df['sentence'].to_list()
    test_preds = predict_classes(counts, test_sentences)
    
    print("Evaluate model performance using f1 metrics", flush=True)
    #Evaluate model performance using f1 metrics
    f1 = eval_bae(test_truths,test_preds)
    f1.to_csv(os.path.join(output_dir, "test_metrics.csv"), sep='\t', index_label="Averaging")
    
    
    
    ### Evaluate the special test file if it exists
    if test_file:
        print("Evaluating secondary test file", flush=True)
        special_test_df = pd.read_csv(test_file,index_col=0)
        special_test_df['sentence'] = po_sentences(special_test_df.index.to_list(), test_motifs)
        special_test_df['sentence'].to_csv(os.path.join(output_dir,"extra_test_sentences.csv"))
        special_test_truths = special_test_df[LABEL_COL].to_list()
        special_test_sentences = special_test_df['sentence'].to_list()
        special_test_preds = predict_classes(counts, special_test_sentences)
        special_f1 = eval_bae(special_test_truths, special_test_preds)
        special_name = test_file.split('/')[-1].split('.')[0]
        special_f1.to_csv(os.path.join(output_dir,special_name+"_test_metrics.csv"), sep='\t', index_label="Averaging")
    
    print("Finished", flush=True)
    #bye bye


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_file", type=str, help="Should be a pd Dataframe.  Needs to have a sequence, label, and partition column to split between training and testing")
    parser.add_argument("motif_file", type=str, help="Path to parquet that stores pre-scanned motifs")
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    #parser.add_argument("sentence_key", type=str, help='Key to the function that converts raw sequence to sentences')
    parser.add_argument("--test_file", type=str, default=None, help="Additional file path for testing. Should be a pandas dataframe with a sequence and label column")
    parser.add_argument("--test_motifs", type=str, default=None, help="Path to additional parquet that stores pre-scanned motifs")
    args = parser.parse_args()
    
    main(**vars(args))










