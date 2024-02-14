#Basic module to convert sequences to sentences
# Daniel Lyon | Cohen Lab| 1/26/24

import numpy as np
import pandas as pd
import math




def sentences_po(labels:list, file_path:str, seq_len:int = 164, k:int=5, divs:int=4) -> list:
    
    print(len(labels), "sentences to create", flush=True)
    
    motif_df = pd.read_parquet(file_path)
    
    score_interval = (1.05 - min(motif_df['occupancy']))/divs
    sentences = []
    
    if type(labels != list):
        labels = list(labels)
    
    ####
    tracker = 0
    q = math.floor(len(labels)/4)
    ####
    existing = set(motif_df.index.get_level_values(0))
    
    for label in labels:
        if label not in existing:
            sentences.append(str(math.ceil(seq_len/k)))
            continue
        
        sdf = motif_df.loc[label]
        i = 0
        s = []
        for loc, row in sdf.iterrows():
            d = loc - i
            if d > 0:
                s.append(str(math.ceil(d/k)))
            s.append(
                "-".join(
                    [row['motif'],
                    row['strand'],
                    str(int(row['occupancy']/score_interval))]))
            i = loc + row['length'] - 1
        s.append(
            str(math.ceil(
                (seq_len - 1 - i)/k
            ))
        )
        sentences.append(" ".join(s))
        
        ###
        tracker += 1
        if tracker % q == 0:
            print('\t', tracker,'/',len(labels), "sentences created", flush=True)
        
        
    print("All sentences created", flush=True)
    return sentences

def sentences_allosteric(labels:list, file_path:str, seq_len:int = 164, k:int=5, divs:int=4, allostery:int=10) -> list:
    
    print(len(labels), "sentences to create", flush=True)
    
    motif_df = pd.read_parquet(file_path)
    
    score_interval = (1 - min(motif_df['occupancy']))/divs
    sentences = []
    
    if type(labels != list):
        labels = list(labels)
    
    ####
    tracker = 0
    q = math.floor(len(labels)/4)
    ####
    existing = set(motif_df.index.get_level_values(0))
    
    for label in labels:
        if label not in existing:
            sentences.append(str(math.ceil(seq_len/k)))
            continue
        
        sdf = motif_df.loc[label]
        i = 0
        s = []
        for loc, row in sdf.iterrows():
            d = loc - i
            if d > 0:
                if d <=allostery and i !=0:
                    prior_motif = s[-1].split('-')[0]
                    del s[-1]
                    s.append(prior_motif+'-'+row['motif'])
                else:
                    s.append(str(math.ceil(d/k)))
            s.append(
                "-".join(
                    [row['motif'],
                    row['strand'],
                    str(int(row['occupancy']/score_interval))]))
            i = loc + row['length'] - 1
        s.append(
            str(math.ceil(
                (seq_len - 1 - i)/k
            ))
        )
        sentences.append(" ".join(s))
        
        ###
        tracker += 1
        if tracker % q == 0:
            print('\t', tracker,'/',len(labels), "sentences created", flush=True)
        
        
    print("All sentences created", flush=True)
    return sentences


def sentences_Cspacer(labels:list, file_path:str, seq_len:int = 164, k:int=5, divs:int=4) -> list:
    
    print(len(labels), "sentences to create", flush=True)
    
    motif_df = pd.read_parquet(file_path)
    
    score_interval = (1.05 - min(motif_df['occupancy']))/divs
    spacer_cut = math.floor(k/2)
    sentences = []
    
    if type(labels != list):
        labels = list(labels)
    
    ####
    tracker = 0
    q = math.floor(len(labels)/4)
    ####
    existing = set(motif_df.index.get_level_values(0))
    
    for label in labels:
        if label not in existing:
            num_spaces = math.floor(seq_len/k)
            if seq_len % k > spacer_cut:
                num_spaces += 1
            sentences.append(" ".join(['SPACE']*num_spaces))    
            continue
        
        sdf = motif_df.loc[label]
        i = 0
        s = []
        for loc, row in sdf.iterrows():
            d = loc - i
            if d > 0:
                num_spaces = math.floor(d/k)
                if d % k > spacer_cut:
                    num_spaces += 1
                if num_spaces > 0:
                    s += ['SPACE'] * num_spaces
            s.append(
                "-".join(
                    [row['motif'],
                    row['strand'],
                    str(int(row['occupancy']/score_interval))]))
            i = loc + row['length'] - 1
        d = seq_len-1-i
        num_spaces = math.floor(d/k)
        if d % k > spacer_cut:
            num_spaces += 1
        if num_spaces > 0:
            s += ['SPACE'] * num_spaces
        
        sentences.append(" ".join(s))
        
        ###
        tracker += 1
        if tracker % q == 0:
            print('\t', tracker,'/',len(labels), "sentences created", flush=True)
        
        
    print("All sentences created", flush=True)
    return sentences
    
def clean_sentences(labels:list, file_path:str, seq_len:int = 164, k:int=5) -> list:
    
    print(len(labels), "sentences to create", flush=True)
    
    motif_df = pd.read_parquet(file_path)
    
    spacer_cut = math.floor(k/2)
    sentences = []
    
    if type(labels != list):
        labels = list(labels)
    
    ####
    tracker = 0
    q = math.floor(len(labels)/4)
    ####
    existing = set(motif_df.index.get_level_values(0))
    
    for label in labels:
        if label not in existing:
            num_spaces = math.floor(seq_len/k)
            if seq_len % k > spacer_cut:
                num_spaces += 1
            sentences.append(" ".join(['SPACE']*num_spaces))    
            continue
        
        sdf = motif_df.loc[label]
        i = 0
        s = []
        for loc, row in sdf.iterrows():
            d = loc - i
            if d > 0:
                num_spaces = math.floor(d/k)
                if d % k > spacer_cut:
                    num_spaces += 1
                if num_spaces > 0:
                    s += ['SPACE'] * num_spaces
            s.append(row['motif'])
            i = loc + row['length'] - 1
        d = seq_len-1-i
        num_spaces = math.floor(d/k)
        if d % k > spacer_cut:
            num_spaces += 1
        if num_spaces > 0:
            s += ['SPACE'] * num_spaces
        
        sentences.append(" ".join(s))
        
        ###
        tracker += 1
        if tracker % q == 0:
            print('\t', tracker,'/',len(labels), "sentences created", flush=True)
        
        
    print("All sentences created", flush=True)
    return sentences