from numpy.random import default_rng
from fitting.crossvalidation import frrsa
import scipy.io
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.append('/home/hhansen/decon/decon_env/DecontextEmbeddings')
import os 
EMBEDDING_DATA_DIR = '/home/hhansen/decon/decon_env/data'
os.environ['EMBEDDING_DATA_DIR'] = EMBEDDING_DATA_DIR
os.environ['EMBEDDING_EVALUATION_DATA_PATH'] = '/home/hhansen/decon/decon_env/DecontextEmbeddings/helpers/embedding_evaluation/data/'
DATA_DIR = '/home/hhansen/decon/decon_env/DecontextEmbeddings/data'
os.environ['DATA_DIR'] = DATA_DIR
import pickle 

from helpers.data import load_behav, load_sorting, match_behv_sim, yield_static_data
from helpers.intersection import get_intersection_words
from helpers.things_evaluation.evaluate import read_embeddings
from collections import defaultdict

def run_frrsa(predictor, target):
    print(f'Shape of target SPoSE RDM: {target.shape}')

    # Set the main function's parameters.
    preprocess = True
    nonnegative = False
    measures = ('dot', 'cosine_sim')

    predictor = predictor.T
    print(f'Shape of predictor matrix for layer {layer}: {predictor.shape}')
    scores, predicted_matrix, betas, predictions = frrsa(target,
                                                         predictor,
                                                         preprocess,
                                                         nonnegative,
                                                         measures,
                                                         cv=[5,5],
                                                         hyperparams=None,
                                                         score_type='pearson',
                                                         wanted=['predicted_matrix'],
                                                         parallel='20',
                                                         random_state=None)
    print(scores)
    reweighted_score = scores.loc[0, 'score']
    return reweighted_score, predicted_matrix



matching_things_ids_word = get_intersection_words(None, 'wikidumps', 'main_word', folder='thinga')
matching_things_ids_synset = get_intersection_words(None, 'wikidumps', 'synset', folder='thinga')

matching_things_ids = set(matching_things_ids_word).intersection(set(matching_things_ids_synset))
print(len(matching_things_ids))

behav_sim = load_behav()
sorting_df = load_sorting()
target_similarity_matrix = match_behv_sim(behav_sim, matching_things_ids, sorting_df)

with open(f'spose_similarity.pkl', 'wb') as o_file:
    pickle.dump(target_similarity_matrix, o_file)

results_word = {}
results_synset = {}
results_static = {}

combs = {
    'bert-base': range(13),
    'gpt-2': range(13),
    'bert-large': range(25),
    'gpt-2-medium': range(25)
}


for matching, result in [('main_word', results_word), ('synset', results_synset)]:
    for model, layers in combs.items():
        if model not in result:
            result[model] = []
            
        for layer in layers:
            print(f'{model} {layer}')
            path = f'{EMBEDDING_DATA_DIR}/thinga/wikidumps/decontext/{model}/{layer}/{matching}/mean/all/decontext.txt'
            embedding_df = read_embeddings(path, matching, matching_things_ids, None, False)
            reweighted_correlation, predicted_matrix = run_frrsa(embedding_df, target_similarity_matrix)
            result[model].append((reweighted_correlation, predicted_matrix))
    
    with open(f'frrsa_results_{matching}.pkl', 'wb') as o_file:
        pickle.dump(result, o_file)

for model, path, static_matching in yield_static_data('thinga'): 
    embedding_df = read_embeddings(path, static_matching, matching_things_ids, None, False)
    reweighted_correlation, predicted_matrix = run_frrsa(embedding_df, target_similarity_matrix)
    results_static[model] = (reweighted_correlation, predicted_matrix)


with open(f'frrsa_results_static.pkl', 'wb') as o_file:
    pickle.dump(results_static, o_file)
