from helpers.data import yield_static_data, load_embedding_to_df
import pandas as pd 
import os
from os.path import join as pjoin

DATA_DIR = os.getenv('DATA_DIR')

def get_word_id(df, matching):
    if matching == 'word':
        df['word'] = df['word'].apply(lambda word: word.lower())
    elif matching == 'synset':
        mapping = pd.read_csv(pjoin(DATA_DIR,'embeddings/data/word_sim/wordsim_synsets.csv'), sep=',')
        df = df.merge(mapping, left_on=matching, right_on='synset')
        df = df.drop(['synset'], axis=1)
    return df
        

def read_wordsim_embeddings(embedding_path, matching_words=[], min_n_contexts=None, matching=None, as_df=False):
    df = load_embedding_to_df(embedding_path, matching)

    if min_n_contexts:
        df = df[df['n_contexts'] >= min_n_contexts]

    df = get_word_id(df, matching)

    df = df.set_index('word')   

    if matching_words:
        df = df.loc[df.index.isin(matching_words)]

    df = df.drop(['n_contexts'], axis=1)

    if not as_df:    
        embeddings = {}
        for row in df.iterrows():
            embeddings[row[0]] = row[1].values
        return embeddings
    return df



