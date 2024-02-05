# Takes in any dataframe containing sequences, IDs, and a .meme file.  Outputs a motif file with motif, direction, start-length coords, and predicted occupancy scores
# Daniel Lyon | WUSTL Cohen Lab | Jan, 2024

import os
import argparse
import pandas as pd
from mpra_tools.predicted_occupancy import *

SEQUENCE_COL = 'sequence'
    
def main(data_file, pwm_file, output_file, cutoff, mu):
    
    output_dir = output_file.split('/')[:-1]
    os.makedirs('/'.join(output_dir), exist_ok=True)
    
    # Load in data file as pandas df
    data_df = pd.read_csv(data_file,index_col=0)
    
    # Read pwm file and convert pwm to energy weight matrix
    ewm = read_pwm_to_ewm(pwm_file)
    
    # Create a motif dataframe with sequence ID and start pos as indeces
    # motif, orientation, length, and occupancy as rows
    index_names = ['label', 'position']
    columns = ['motif', 'strand', 'length', 'occupancy']
    
    sequences = data_df['sequence']
    index = []
    data = []
    
    print(len(sequences), 'sequences to scan', flush=True)
    counter = 0
    for label, seq in sequences.items():
        landscape = total_landscape(seq, ewm, mu)
        for loc, motifs in landscape.iterrows():
            for id in landscape.columns:
                if motifs[id]>cutoff:
                    index.append((label,loc))
                    data.append([id.split('_')[0],id[-1],len(ewm[id[:-2]]) ,motifs[id]])
        counter += 1
        if counter % 2000 == 0:
            print('\t', counter, '/', len(sequences), 'sequences scanned', flush=True)
    
    motifs_df = pd.DataFrame(
        data = data,
        index = pd.MultiIndex.from_tuples(index, names=index_names),
        columns=columns
    )
    
    #Save motif file to destination
    if(output_file.split('.')[-1] == 'parquet'):
        motifs_df.to_parquet(output_file)
    else:
        motifs_df.to_csv(output_file)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_file", type=str, help="Should be a pd Dataframe.  Needs to have a sequence")
    parser.add_argument("pwm_file", type=str, help="Path to .meme file that stores motif PWMs")
    parser.add_argument("output_file", type=str, help='Full name and dir of the file')
    parser.add_argument("--cutoff", type=float, default=0.2, help="The minimum predicted occupancy a motif requires to be collected")
    parser.add_argument("--mu", type=int, default=9, help="Predicted affinity of TF to motif")
    
    args = parser.parse_args()
    
    main(**vars(args))










