#Basic module to convert sequences to sentences
# Daniel Lyon | Cohen Lab| 1/26/24

import numpy as np
import pandas as pd
import math




def po_sentences(labels:list[str], file_path:str, seq_len:int = 164, k:int=5, divs:int=4) -> list[str]:
    
    print(len(labels), "sentences to create", flush=True)
    
    motif_df = pd.read_parquet(file_path)
    
    score_interval = (1 - min(motif_df['occupancy']))/divs
    sentences = []
    
    if type(labels != list):
        lables = list(labels)
    
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
            
        
    
    
    
    

