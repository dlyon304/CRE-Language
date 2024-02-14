import os
import sys
import argparse
import numpy as np
import pandas as pd
import random

def check_test(df, frac=0.15):
    if 'set' not in df.columns:
        df['set'] = random.choices(['TRAIN','TEST'], cum_weights=[1-frac,1], k=len(df))
        print("No Training / Test set split found in datafile. Setting dedicated test set at:", frac*100, "percent", flush=True)
    
    num_test = df['set'].value_counts()['TEST']
    if num_test/len(df)<0.1 or num_test/len(df)>0.3:
        df['set'] = random.choices(['TRAIN','TEST'], cum_weights=[1-frac,1], k=len(df))
        print("Test set found to be below 10% or above 30%. Setting dedicated test set at:", frac*100, "percent", flush=True)
    

def check_folds(df, n_folds):
    if 'fold' not in df.columns:
        df['fold'] = random.choices(range(1,n_folds+1), k=len(df))
        print("No cross validation fold key ('fold') found in datafile.  Creating", n_folds, "of cross-folds", flush=True)
    if n_folds < 2:
        print("FATAL: Less than 2 validation folds specified.  Please write validation folds into file under column 'fold'", flush=True)
        raise ValueError("FATAL: Less than 2 validation folds specified.  Please write validation folds into file under column 'fold'")
    
    expected = set(range(1,n_folds+1))
    observed = set(df['fold'].values)
    if expected == observed:
        return
    print("Number of folds specified does not match up with folds labeled in the datafile. Fixing so that there are the correct number of folds", flush=True)
    df['fold'] = random.choices(range(1,n_folds+1), k=len(df))
    

if __name__ == "__main__":
    args = sys.argv
    datafile_path = args[1]
    num_folds = int(args[2])
    
    filename, file_extension = os.path.splitext(datafile_path)
    if file_extension == 'parquet':
        data_df = pd.read_parquet(datafile_path)
    else:
        data_df = pd.read_csv(datafile_path, index_col=0)
    
    check_test(data_df)
    check_folds(data_df, num_folds)
    
    if file_extension == 'parquet':
        data_df.to_parquet(datafile_path)
    else:
        data_df.to_csv(datafile_path)
    
    
    






