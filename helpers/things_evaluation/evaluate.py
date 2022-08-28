import argparse

import numpy as np 
import pandas as pd
import scipy.io
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os 
from copy import deepcopy
from os.path import join as pjoin 

from helpers.data import load_embedding_to_df, load_things_mapping

DATA_DIR = os.getenv('DATA_DIR')
print(DATA_DIR)

def create_cosine_similarity_vector(df, model):
    #pred_sim_matrix = 1 - euclidean_distances(df, df)
    pred_sim_matrix = cosine_similarity(df, df)
    sim_df_matrix = pd.DataFrame(pred_sim_matrix, columns=df.index, index=df.index)
    sim_vector = squareform(pred_sim_matrix, force='tovector', checks=False)
    return sim_df_matrix, sim_vector

def load_behav():
    # Read similarity data from THINGS behaviourial data
    behv_sim = scipy.io.loadmat(pjoin(DATA_DIR,'things/spose_similarity.mat'))['spose_sim']
    return behv_sim

def read_embeddings(embedding_path, matching, matching_words=None, min_n_contexts=None, keep_n_contexts=False, keep_categories=False):
    df = load_embedding_to_df(embedding_path, matching)

    if min_n_contexts:
        df = df[df['n_contexts'] >= min_n_contexts]

    if not keep_n_contexts:
        df = df.drop(['n_contexts'], axis=1)

    df = get_things_id(df, matching)
    
    if matching_words:
        df = df[df['things_id'].isin(matching_words)]

    # remove embeddings without matching synset
    tdf = pd.read_csv(pjoin(DATA_DIR,'things/things_concepts.tsv'), sep='\t')

    if matching == 'synset':
        things_synsets = list(tdf['Wordnet ID2'])
        df = df[df['synset'].isin(things_synsets)]
    elif matching == 'word' or matching == 'main_word':
        things_ids = list(tdf['uniqueID'])
        df = df[df['things_id'].isin(things_ids)]
    elif matching == 'concept_id':
        things_synsets = list(tdf['Wordnet ID4'])
        df = df[df['concept_id'].isin(things_synsets)]
    
    df = df.set_index('things_id')

    

    # TODO only embedding where ares was used -> anaylsis if synsets make sense
   # if only_ares_matched and matching == 'synset':
    #    df = df[df['synset'] != 'no_ares_rep']
    
    df = sort_df(df, load_sorting())

    df = df.drop([matching], axis=1)
    

    if keep_categories:
        things_df = load_things_data().rename(columns={"Bottom-up Category (Human Raters)": 'category', 'uniqueID': 'things_id'})[['category', 'things_id']]
        def cut(cat):
            if type(cat) == str:
                return cat.split(',')[0]
            return cat
        things_df['category'] = things_df['category'].apply(cut)
        things_df.set_index('things_id')
        df = things_df.merge(df, on='things_id')
        df = df.drop(['things_id'], axis=1)
    return df

def load_things_data():
    things_df = pd.read_csv(pjoin(DATA_DIR,'thinga/things_concepts.tsv'), sep='\t')
    return things_df

def load_sorting():
    # Read things concept sorting
    sorting_df = pd.read_csv(pjoin(DATA_DIR,'things/unique_id.txt'), header=None, names=['concept_id'])
    return sorting_df


def format_concept(row, synset_row, matching):
    things_df = pd.read_csv(pjoin(DATA_DIR,'things/things_concepts.tsv'), sep='\t')
    synset = row[matching]
    concept = things_df[things_df[synset_row] == synset]['uniqueID']
    if not concept.empty:
        concept = concept.iloc[0]
    else:
        concept = np.nan
    return concept 


def get_things_id(df, matching):
    if matching == 'synset':
        synset_row = 'Wordnet ID2'
        df['things_id'] = df.apply(lambda row: format_concept(row, synset_row, matching), axis=1)
    elif matching == 'concept_id':
        synset_row = 'Wordnet ID4'
        df['things_id'] = df.apply(lambda row: format_concept(row, synset_row, matching), axis=1)
    else:
        rows = []
        syn_df = load_things_mapping()
        for row in df.iterrows():
            row = row[1]
            word = row[matching]
            ids = syn_df.loc[syn_df['Word'] == word]['id']
            for id in ids:
                row['things_id'] = id
                rows.append(deepcopy(row))
        df = pd.DataFrame(rows)

    return df

def match_behv_sim(behv_sim, sorting_df, concepts_to_keep):
    concept_positions_to_keep = [sorting_df.index[sorting_df['concept_id'] == concept].tolist()[0] for concept in concepts_to_keep]
    concept_positions_to_keep = sorted(concept_positions_to_keep)
    behv_sim_matched = behv_sim[concept_positions_to_keep, :]
    behv_sim_matched = behv_sim_matched[:, concept_positions_to_keep]
    return behv_sim_matched

def sort_df(df, sorting_df):
    sorted_df = sorting_df.reset_index().set_index('concept_id')
    df['concept_num'] = df.index.map(sorted_df['index'])
    df = df.sort_values(by='concept_num')
    df = df.drop('concept_num', axis=1)
    return df

def plot_behav_sim(behav_sim):
    plot = sns.clustermap(behav_sim, yticklabels=10, xticklabels=10)
    plot.fig.suptitle('THINGS') 
    plt.savefig(f"behav_sim.png")
    plt.clf()
    
def evaluate(embeddings, matching, model='bla', matching_words=None, min_n_contexts=None):
    df = None
    if type(embeddings) == str:
        df = read_embeddings(embeddings, matching, matching_words, min_n_contexts)
    else:
        df = embeddings

    print(f'number of rows: {df.index}')
    behav_sim = load_behav()
    sorting_df = load_sorting()
    
    things_concepts_to_keep = list(df.index)
    behav_sim = match_behv_sim(behav_sim, sorting_df, things_concepts_to_keep)
    #plot_behav_sim(behav_sim)
    behav_sim = squareform(behav_sim, force='tovector', checks=False)
    sim_df_matrix, sim_vector = create_cosine_similarity_vector(df, model)

    pearson = np.corrcoef(behav_sim, sim_vector)[1][0]
    spearman = spearmanr(behav_sim, sim_vector)
    return pearson, spearman, sim_df_matrix, sim_vector

    

    