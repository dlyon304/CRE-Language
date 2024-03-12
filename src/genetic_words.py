# Genetic search for words
# Daniel Lyon | WUSTL Cohen Lab | Jan, 2024

import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.metrics import f1_score
from grammar import sentences
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
        [word_counter.update(s) for s in sentences.to_list()]
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
    [alphabet.update(s) for s in test_sentences]
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
            preds[c] += sum(class_word_logs[c][word] for word in sentence)
        
        sentence_log_preds.append(preds.idxmax())
    
    return sentence_log_preds

def eval_bae(Ytrue: list[str], Ypred: list[str]) -> float:
    
    classes = list(set(Ytrue))
    return f1_score(Ytrue, Ypred, labels=classes, average='weighted')
    
def cut_sentences(w, words):
    if w in words:
        return w
    else:
        return "SPACE"
    
    
    
def main(data_file, motif_file, output_dir, initial_size, N, generations):
    
    print("Read in data", flush=True)
    sentences = pd.read_csv("Data/Sentences/all_sentences.csv", index_col=0)
    sentences = sentences['sentence'].map(lambda x: np.array(x.split()))
    vCut = np.vectorize(cut_sentences)
    data_df = pd.read_csv(data_file,index_col=0)
    data_df['split'] = random.choices([TRAIN_KEY, TEST_KEY], cum_weights=[0.8,1], k=len(data_df))

    print("Split train and test from main dataframe", flush=True)
    #Split train and test from main dataframe
    train_i = data_df[data_df[SPLIT_COL]==TRAIN_KEY].index
    test_i = data_df[data_df[SPLIT_COL]==TEST_KEY].index
    
    print("Creating Universe")
    motif_df = pd.read_parquet(motif_file)
    all_motifs = list(set(motif_df['motif']))
    del motif_df
    print(len(all_motifs), flush=True)
    universe = dict([(i,random.sample(all_motifs, k=initial_size)) for i in range(N)])
    fitness = dict([(i,0) for i in range(N)])
    
    for generation in range(generations):
        
        for index, word_list in universe.items():
            word_list_set = set(word_list)
            index_sentences = sentences.copy()
            index_sentences = index_sentences.map(lambda x: vCut(x, word_list_set))
            data_df['sentence'] = index_sentences
            
            counts, priors = get_counts(data_df.loc[train_i].groupby(LABEL_COL)['sentence'])
            test_truths = data_df.loc[test_i][LABEL_COL].to_list()
            test_sentences = data_df.loc[test_i]['sentence'].to_list()
            test_preds = predict_classes(counts, test_sentences)
            fitness[index] = eval_bae(test_truths,test_preds)
            print(fitness[index], flush = True)
            print(word_list, flush=True)
        
        print("Tested fitness for generation", generation)
        #OK we have the fitness for each WordList now hopefully
        #sort them
        #keep the best half, delete the worst half
        #randomly mutate 2
        #continue
        pairs = sorted(fitness.items(), key=lambda x: x[1], reverse=True)
        
        cutoff = int(N/2)
        mutate = 2
        parents = random.choices(pairs[:cutoff], k=cutoff)
        for n in range(cutoff):
            # mutate
            pm = random.sample(range(len(parents[n])), k=mutate)
            child = universe[parents[n][0]].tolist()
            for point in pm:
                child[point] = random.choice(all_motifs)
                
             # add / del
            k = random.random()
            if k < 0.1:
                child.append(random.choice(all_motifs))
            elif k < 0.2:
                del child[random.choice(len(child))]
            universe[pairs[cutoff+n][0]] = child
        
        #output 10 highest scores
        print('----------------------', generation, flush=True)
        print("Average:", np.mean([x[1] for x in pairs]))
        for i in range(10):
            print(pairs[i][1])
            print(universe[pairs[1][0]])     
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_file", type=str, help="Should be a pd Dataframe.  Needs to have a sequence, label, and partition column to split between training and testing")
    parser.add_argument("motif_file", type=str, help="Path to parquet that stores pre-scanned motifs")
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    #parser.add_argument("sentence_key", type=str, help='Key to the function that converts raw sequence to sentences')
    parser.add_argument("--initial_size", type=int, default=10, help="Size of initital word lists")
    parser.add_argument("--N", type=int, default=10, help="Number of word lists")
    parser.add_argument("--generations", type=int, default = 20, help="Number of generations")
    args = parser.parse_args()
    
    main(**vars(args))










