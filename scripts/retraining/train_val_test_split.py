from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 

import sys 
sys.path.append('../..')

from helpers.things_evaluation.evaluate import load_behav, load_sorting, match_behv_sim

def load_things():
    behav_sim = load_behav()
    sorting_df = load_sorting()
    things_df = pd.DataFrame(behav_sim, index=sorting_df['concept_id'], columns=sorting_df['concept_id'])
    things_df = things_df.reindex(sorted(things_df.columns), axis=1)
    things_df = things_df.reindex(sorted(things_df.columns), axis=0)
    return things_df

def train_words(things):
    # THINGs word that shall be used
    words = list(things.index)

    # dont use homonyms for fine tuning 
    words = [word for word in words if not any(char.isdigit() for char in word)]
    words.remove('iceskate')
    words.remove('ticktacktoe')

    train_words, test_words = train_test_split(np.array(words), test_size=0.6, random_state=42)
    train_words, val_words = train_test_split(np.array(train_words), test_size=0.3, random_state=42)
    
    with open('train_words.txt', 'w') as o_file:
        for word in train_words:
            o_file.write(word)
            o_file.write('\n')

    with open('test_words.txt', 'w') as o_file:
        for word in test_words:
            o_file.write(word)
            o_file.write('\n')

    with open('val_words.txt', 'w') as o_file:
        for word in val_words:
            o_file.write(word)
            o_file.write('\n')

things = load_things()
train_words = train_words(things)