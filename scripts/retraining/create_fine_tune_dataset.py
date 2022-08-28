from collections import defaultdict
from email.policy import default
import statistics
import torch
import pandas as pd
import random
import numpy as np 
import matplotlib.pyplot as plt 

import os 
os.environ['DATA_DIR'] = '/home/hannes/data/laptop_sync/Uni/master/masterarbeit/decon/DecontextEmbeddings/data'
os.environ['EMBEDDING_EVALUATION_DATA_PATH'] = './embedding_evaluation/data/'

import sys 
sys.path.append('../..')

from helpers.things_evaluation.evaluate import load_behav, load_sorting, match_behv_sim

def load_embeddings(embeddings_path):
    embeddings = defaultdict(list)
    with open(embeddings_path, 'r') as embeddings_file:
        for line in embeddings_file:
            line = line.strip()
            split = line.split(';')
            word = split[0]

            # word to things_id
            word = word.replace(' ', '_')
            #embedding = split[2].split(' ')
            #embedding = torch.as_tensor([float(value) for value in embedding])
            embeddings[word].append(split[2])
    return embeddings


def load_things():
    behav_sim = load_behav()
    sorting_df = load_sorting()
    things_df = pd.DataFrame(behav_sim, index=sorting_df['concept_id'], columns=sorting_df['concept_id'])
    things_df = things_df.reindex(sorted(things_df.columns), axis=1)
    things_df = things_df.reindex(sorted(things_df.columns), axis=0)
    return things_df

def create(train_words, things):
    similarity_pairs = defaultdict(list)
    pairs = []
    n_embeddings_per_pair = 1
    embeddings = load_embeddings('../../../data_fine_tune/extractions.txt')
    #train_words = ['dog', 'chipmunk', 'toaster']
    sims = []

    for i, word1 in enumerate(train_words):
        # also similar word pairs like cat - cat to get pairs with similarity 1
        for word2 in train_words[i+1:]:
            similarity = things.loc[word1, word2]
            for i in range(n_embeddings_per_pair):
                embedding1 = random.choice(embeddings[word1])
                embedding2 = random.choice(embeddings[word2])
                #similarity = round(similarity, 1)
                #similarity_pairs[similarity].append((word1, word2, embedding1, embedding2))
                pair = (embedding1, embedding2, str(similarity))
                pairs.append(pair)
                sims.append(similarity)

    #for i in range(100):
    #    similarity = random.choice(list(similarity_pairs.keys()))
    #    random_emb = random.choice(similarity_pairs[similarity])
    #    pair = (random_emb[1], random_emb[2], random_emb[3], str(similarity))
    #pairs.append(pair)
    print(statistics.mean(sims))
    return pairs

def write(pairs, name):
    with open(f'{name}_fine_tune_dataset.csv', 'w') as o_file:
        for embedding1, embedding2, similarity_str in pairs:
            o_file.write(';'.join([embedding1, embedding2, similarity_str]))
            o_file.write('\n')        
            

things = load_things()

train_words = pd.read_csv('./train_words.txt', header=None, names=['word'])['word']
pairs = create(train_words, things)
write(pairs, 'train')

val_words = pd.read_csv('./val_words.txt', header=None, names=['word'])['word']
pairs = create(val_words, things)
write(pairs, 'val')


