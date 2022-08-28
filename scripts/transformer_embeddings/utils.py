#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
import re
import torch

import logging
import numpy as np
import pandas as pd

from numba import njit, prange
from collections import defaultdict
from os.path import join as pjoin
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from itertools import chain
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator

main_logger = logging.getLogger('main')

#################################################################################################################
############################## HELPERS FOR LOADING MODELS AND TOKENIZERS INTO MEMORY ############################
#################################################################################################################


def load_model(model_name: str, device: torch.device, output_hiddens: bool = True):
    """load pretrained LM into memory"""
    if model_name == 'distilbert':
        from transformers import DistilBertModel
        pretrained_weights = 'distilbert-base-uncased'
        model = DistilBertModel.from_pretrained(
            pretrained_weights,
            return_dict=True,
            output_hidden_states=output_hiddens,
        )
    elif model_name == 'sbert_bert':
        from transformers import AutoModel
        name = 'sentence-transformers/bert-base-nli-mean-tokens'
        model = AutoModel.from_pretrained(name, return_dict=True, output_hidden_states=output_hiddens)
    elif re.search(r'^bert', model_name):
        from transformers import BertModel
        pretrained_weights = 'bert-large-uncased' if re.search(
            r'large$', model_name) else 'bert-base-uncased'
        model = BertModel.from_pretrained(
            pretrained_weights,
            return_dict=True,
            output_hidden_states=output_hiddens,
            local_files_only=True
        )
    elif re.search(r'^funnel', model_name):
        from transformers import FunnelModel
        pretrained_weights = 'funnel-transformer/large'
        model = FunnelModel.from_pretrained(
            pretrained_weights,
            return_dict=True,
            output_hidden_states=output_hiddens,
        )
    elif re.search(r'^gpt', model_name):
        from transformers import GPT2Model
        pretrained_weights = 'gpt2-medium' if re.search(
            r'medium$', model_name) else 'gpt2'
        model = GPT2Model.from_pretrained(
            pretrained_weights,
            return_dict=True,
            output_hidden_states=output_hiddens,
        )
    else:
        from transformers import XLNetModel
        pretrained_weights = 'xlnet-large-cased'
        model = XLNetModel.from_pretrained(
            pretrained_weights,
            return_dict=True,
            output_hidden_states=output_hiddens,
        )
    model.eval()
    model.to(device)
    return model


def load_tokenizer(model_name: str):
    """load tokenizer corresponding to pretrained LM into memory"""
    if model_name == 'distilbert':
        from transformers import DistilBertTokenizerFast
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            pretrained_weights, padding_side='right')
    elif model_name == 'sbert_bert':
        from transformers import AutoTokenizer
        #name = 'sentence-transformers/all-distilroberta-v1'
        name = 'sentence-transformers/bert-base-nli-mean-tokens'
        tokenizer = AutoTokenizer.from_pretrained(name, padding_side='right', model_max_length=512)
    elif re.search(r'^bert', model_name):
        from transformers import BertTokenizerFast
        pretrained_weights = 'bert-large-uncased' if re.search(
            r'large$', model_name) else 'bert-base-uncased'
        tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_weights, padding_side='right', model_max_length=512,local_files_only=True)
    elif re.search(r'^funnel', model_name):
        from transformers import FunnelTokenizerFast
        pretrained_weights = 'funnel-transformer/large'
        tokenizer = FunnelTokenizerFast.from_pretrained(
            pretrained_weights, padding_side='right')
    elif re.search(r'^gpt', model_name):
        from transformers import GPT2TokenizerFast
        pretrained_weights = 'gpt2-medium' if re.search(
            r'medium$', model_name) else 'gpt2'
        tokenizer = GPT2TokenizerFast.from_pretrained(
            pretrained_weights, padding_side='right', add_prefix_space=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        from transformers import XLNetTokenizerFast
        pretrained_weights = 'xlnet-large-cased'
        tokenizer = XLNetTokenizerFast.from_pretrained(
            pretrained_weights, padding_side='right')
    return tokenizer

###########################################################################
############################## GENERAL HELPERS ############################
###########################################################################


def csv_to_list(file_name, sep=',', column=None):
    df = pd.read_csv(file_name, sep=sep)
    if column is not None and column in df.columns:
        return list(df[column])
    return [list(df[c] for c in df.columns)]


def file_to_lines(f):
    with open(f, 'r') as input_file:
        main_logger.debug(f'read file {f}')
        for line in input_file:
            yield line

def logger_thread(q):
    error_logger = logging.getLogger('main')
    error_logger.debug('start logger thread')
    while True:
        record = q.get()
        if record is None:
            break
        error_logger.debug(record)

def unpickle_file(f: str):
    return pickle.loads(open(f, 'rb').read())


def get_things_db(sub_folder: str = 'things', main_folder: str = './data') -> pd.DataFrame:
    return pd.read_csv(pjoin(main_folder, sub_folder, 'item_names.tsv'), encoding='utf-8', sep='\t')


def get_obj_names_and_ids(sub_folder: str = 'things', main_folder: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    obj_data = pd.read_csv(pjoin(main_folder, sub_folder, 'item_names.tsv'),
                           encoding='utf-8', sep='\t')
    obj_names = obj_data.Word.values
    wordnet_ids = obj_data[obj_data.columns[-2]].values
    return obj_names, wordnet_ids


def add_period(sent: str) -> str:
    return ''.join((sent, '.'))


def assert_punctuation(sentences: list) -> List[str]:
    return [add_period(sent) if not re.search(r'[\.?!]$', sent) else sent for sent in sentences]


def prepend_spaces(sentences: list) -> List[str]:
    return [''.join((' ', sent)) for sent in sentences]


def load_reps(PATH: str, sep: str = ' ') -> dict:
    w2v = {}
    with open(PATH, 'r') as f:
        for l in f:
            l = l.strip().split(sep)
            w = l[0]
            v = np.array(list(map(float, l[1:])))
            w2v[w] = v
    return w2v


def get_sense_ids(PATH: str, sep: str = ',') -> list:
    with open(pjoin(PATH, 'words_and_sense_ids.txt'), 'r') as f:
        sense_ids = list(
            map(lambda str: str.strip().split(sep)[1], f.readlines()))
        return sense_ids


def lower_case_words(sent: List[str]) -> List[str]:
    return list(map(lambda w: w.lower(), sent))

def load_embeddings(embeddings_path, synset_word_levels, max_number_of_embeddings):
    counter = defaultdict(lambda: 0)

    embeddings = defaultdict(list)
    for line in file_to_lines(embeddings_path):
        splitted_line = line.split(';')

        word_found_in_sentence = splitted_line[0]
        main_words = splitted_line[1]
        synset = splitted_line[2]
        concept_id = splitted_line[3]
        
        #word_found_in_sentence = splitted_line[0]
        #synset = splitted_line[1]
        
        embedding = splitted_line[4].split(' ')
        embedding = np.asarray([float(value) for value in embedding])
        
        key = None
        if synset_word_levels == 'word':
            keys = [word_found_in_sentence]
        elif synset_word_levels == 'synset':
            keys = [synset]
        elif synset_word_levels == 'concept_id':
            keys = [concept_id]
        elif synset_word_levels == 'main_word':
            keys = main_words.split(',')

        for key in keys:
            if max_number_of_embeddings and counter[key] >= max_number_of_embeddings:
                continue
            
            embeddings[key].append(embedding)
            counter[key] += 1
    
    for key in embeddings:
        embeddings[key] = np.vstack(embeddings[key])

    return embeddings

#####################################################################################################################
###################### HELPERS FOR PROCESSING SENTENCES AND EXTRACTING REPRESENTATIONS ##############################
#####################################################################################################################

def tokenize_sentence(tokenizer, sentence: str, return_offsets: bool, padding: bool=True, truncate: bool = True):
    inputs = tokenizer([sentence], padding=padding, truncation=truncate,
                       return_tensors="pt", return_offsets_mapping=return_offsets, return_special_tokens_mask=True)
        
    return inputs

def process_sentences(tokenizer, sentence, word, model):
    # each sentence needs to be tokenized with batch size 1 without padding and truncation
    processed = tokenize_sentence(tokenizer, sentence, True, False, False).to(model.device)
    sentence_token_ids = processed['input_ids'][0]

    # skip CLS and SEP TOKEN -> shall not be part of the new string
    tokens = tokenizer.convert_ids_to_tokens(sentence_token_ids, skip_special_tokens=True)
    amount_tokens = len(tokens)
    max_number_tokens = tokenizer.model_max_length - 2
    
    if amount_tokens > max_number_tokens:
        sentence = sentence.lower()
        word = word.lower()
        word_pos = sentence.index(word)
        diff = (amount_tokens - max_number_tokens) 
        if word_pos > len(sentence) / 2:
            tokens = tokens[diff:]
        else:
            tokens = tokens[:-diff]
        sentence = tokenizer.convert_tokens_to_string(tokens)
    return sentence

def get_token_ranges(tokenizer, word, processed, logger):        
    indices_of_matched_tokens = []

    for word in [word, ' ' + word]:
        word_tokenized = tokenize_sentence(tokenizer, word, False).to('cpu')
        word_tokens = tokenizer.convert_ids_to_tokens(word_tokenized['input_ids'][0], skip_special_tokens=True)

        #logger.debug(word_tokens_space)
        n_tokens = len(word_tokens)

        tokens_of_one_sentence = tokenizer.convert_ids_to_tokens(processed['input_ids'][0], skip_special_tokens=True)
        #logger.debug(tokens_of_one_sentence)
        indices_of_matched_tokens = []
        for i in range(len(tokens_of_one_sentence)):
            start = None
            end = None
            match = True
            for m in range(n_tokens):
                next_token = tokens_of_one_sentence[i+m]
                if i+m <= len(tokens_of_one_sentence)-1 and next_token == word_tokens[m]:
                    if m == 0:
                        start = i+m
                        if n_tokens == 1:
                            end = i+m
                    elif m == n_tokens-1:
                        end = i+m
                    continue
                else:
                    match = False  
                    start = None 
                    end = None
                    break 

            if match:
                start_end_indices = (start, end)
                indices_of_matched_tokens.append(start_end_indices)
                start = None 
                end = None

    # using word version with space for GPT -> but for BERT will lead to double indices as BERT tokenizes word with and without space same
    return list(set(indices_of_matched_tokens))


def get_embeddings_of_word(word, processed, all_token_embeddings, tokenizer, logger):
    token_ranges = get_token_ranges(tokenizer, word, processed, logger)	
    embeddings = []
    for token_range in token_ranges:
        start_token, end_token = token_range

        # word got split into subwords -> multiple tokens
        if start_token != end_token:
            all_tokens = range(start_token, end_token+1)
            embeddings_of_tokens = all_token_embeddings[all_tokens, :]
            embedding_of_word = embeddings_of_tokens.mean(dim=0)
            embeddings.append(embedding_of_word)
        else:
            embedding_of_word = all_token_embeddings[start_token, :]
            embeddings.append(embedding_of_word)
		#for i, hidden_list in enumerate(pooled_hiddens):
			#word_embeddings_per_sen = [hidden.cpu() for hidden in hidden_list]
    return embeddings, token_ranges

def get_hiddens(model, inputs: dict) -> Tuple[torch.Tensor]:
    filtered_inputs = {k: v for k, v in inputs.items() if k !=
                       'offset_mapping' and k != 'special_tokens_mask'}
    special_token_mask = inputs['special_tokens_mask'][0]
    with torch.no_grad():
        output = model(**filtered_inputs)
        hidden_states = output.hidden_states
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        
        # remove special tokens like CLS
        token_embeddings = token_embeddings[torch.where(special_token_mask == 0)]
        return token_embeddings


def aggregate_hiddens(hiddens: tuple, start_layer_inclusive: int, end_layer_inclusive: int, reduction: str = 'mean') -> torch.Tensor:
    hidden_reps = torch.stack(hiddens[start_layer_inclusive:end_layer_inclusive+1], dim=1)

    if reduction == 'mean':
        avg_rep = hidden_reps.mean(dim=1)
        return avg_rep
    elif reduction == 'max':
        max_rep = hidden_reps.max(dim=1)[0]
        return max_rep


def get_hidden_indices(h, start, end):
    length = end[0] - start[0]
    inds = torch.arange(length, device=h.device).repeat(h.size(0), 1)
    offsets = torch.tensor(start, device=h.device)
    inds += offsets[:, None]
    return h.gather(1, inds.unsqueeze(-1).expand(-1, -1, h.size(2)))


def pool(h, start, end, pooling):
    indexed_hiddens = get_hidden_indices(h, start, end)
    if pooling == 'mean':
        return torch.mean(indexed_hiddens, dim=1)
    elif pooling == 'max':
        return torch.max(indexed_hiddens, dim=1)[0]


def pool_old(h, start, end, pooling):
    def get_mask():
        mask = torch.arange(0, h.size(1)).repeat(len(start), 1).T
        mask = ((mask >= torch.tensor(start)) & (mask < torch.tensor(end))).T
        return mask.to(h.device)

    mask = get_mask(h, start, end)
    if pooling == 'mean':
        return torch.sum(h * mask.unsqueeze(dim=-1), dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=-1)
    elif pooling == 'max':
        neg_inf_mask = ~mask * -float("Inf")
        neg_inf_mask[torch.isnan(neg_inf_mask)] = 0
        return torch.max(h + neg_inf_mask.unsqueeze(dim=-1), dim=1)[0]


def stack_hiddens(hiddens, ranges):
    return torch.cat([hiddens[i].unsqueeze(0).expand(len(r), -1, -1) for i, r in enumerate(ranges)], 0)


def pool_subword_to_word(hiddens: torch.Tensor, ranges: list, pooling: str = 'mean') -> List[np.ndarray]:
    pooled_hiddens = []
    if ranges and type(ranges[0]) == tuple:
        starts, ends = zip(*ranges)
        pooled_hiddens = pool(hiddens, starts, ends, pooling)
    elif ranges:
        stacked_hiddens = stack_hiddens(hiddens, ranges)
        starts, ends = zip(*[item for sublist in ranges for item in sublist])
        pooled = pool(stacked_hiddens, starts, ends, pooling)
        index = 0
        for r in ranges:
            pooled_hiddens.append(pooled[index:index + len(r)])
            index += len(r)
    return pooled_hiddens


def aggregate_reps(pooled_hiddens: List[torch.Tensor], pooling: str, rnd_seed: int = None, pca_method: str = None, ratio: float = None) -> torch.Tensor:
    """for single word decontextualize pooled hidden representationss across N sentences"""
    if pooling == 'mean':
        decontextualized_rep = pooled_hiddens.mean(axis=0)
    elif pooling == 'max':
        decontextualized_rep = pooled_hiddens.max(axis=0)[0]
    return decontextualized_rep


# TODO: the function below needs to be updated AND tested
def cluster_reps(pooled_hiddens: List[torch.Tensor], k_range: int, specified_k=None) -> torch.Tensor:
    """for single word decontextualize pooled hidden representationss across N sentences"""
    hiddens_per_sent = torch.stack(
        [h for h in pooled_hiddens if isinstance(h, torch.Tensor)]).cpu().numpy()
    all_silhouette_scores = []
    all_sum_of_distances = []
    all_cluster_centers = []

    if not specified_k:
        for n_cluster in range(1, k_range):
            kmeans = KMeans(n_clusters=n_cluster).fit(hiddens_per_sent)
            label = kmeans.labels_
            if n_cluster != 1:
                sil_coeff = silhouette_score(
                    hiddens_per_sent, label, metric='cosine')
                all_silhouette_scores.append(sil_coeff)
            all_cluster_centers.append(kmeans.cluster_centers_)
            all_sum_of_distances.append(kmeans.inertia_)

        # get best cluster centers
        max_value = max(all_silhouette_scores)
        sil_max_index = all_silhouette_scores.index(max_value)

        kn = KneeLocator(list(range(1, k_range)), all_sum_of_distances,
                         curve='convex', direction='decreasing')
        elbow_index = kn.knee

        if not elbow_index:
            max_index = sil_max_index + 1
        elif sil_max_index < elbow_index + 1:
            max_index = sil_max_index + 1
        else:
            max_index = elbow_index

        chosen_centers = all_cluster_centers[max_index]
        return chosen_centers, chosen_centers.shape[0]
    else:
        kmeans = KMeans(n_clusters=specified_k).fit(hiddens_per_sent)
        chosen_centers = kmeans.cluster_centers_
        return chosen_centers, chosen_centers.shape[0]


def pc_mean_elimination_(pca, hiddens: np.ndarray, ratio: float) -> np.ndarray:
    first_pcs = pca.components_[
        np.where(np.cumsum(pca.explained_variance_ratio_) <= ratio)]
    hiddens_mean = hiddens.mean(axis=0)
    anisotropy = np.sum([(pc @ hiddens_mean) * pc for pc in first_pcs], axis=0)
    return hiddens_mean - anisotropy


def remove_top_pcs_(pca, hiddens_pcs: np.ndarray, hiddens: np.ndarray, ratio: float) -> np.ndarray:
    first_pcs = hiddens_pcs[:, np.where(
        np.cumsum(pca.explained_variance_ratio_) <= ratio)[0]]
    anisotropy = np.sum([np.outer(pc, (pc.T @ hiddens))
                        for pc in first_pcs.T], axis=0)
    hiddens_iso = hiddens - anisotropy
    return hiddens_iso.mean(axis=0)

####################################################################################################
############################## HELPERS FOR CORPUS ANNOTATION (WITH ARES) ###########################
####################################################################################################


def cka(x, y) -> float:
    return np.linalg.norm(y @ x)**2 / (np.linalg.norm(x**2) * np.linalg.norm(y**2))


def cosine_sim(x, y) -> float:
    return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))


def nearest_neighbor(point: torch.Tensor, neighbors):
    from scipy.spatial.distance import cosine
    if isinstance(neighbors, dict):
        assert len(point) == len(neighbors[next(iter(neighbors))][0])
        dists = [(k, n[1], cosine(point, n[0])) for k, n in neighbors.items()]
    else:
        assert len(point) == len(neighbors[0])
        dists = [(i, cosine(point, n)) for i, n in enumerate(neighbors)]
    closest_neighbor = min(dists, key=lambda x: x[1])
    return (closest_neighbor[0], closest_neighbor[1])


def syns_to_words(word_set: list, synsets: list) -> dict:
    synset_mapping = defaultdict(list)
    for w, syn in zip(word_set, synsets):
        w = w.replace('_', ' ')
        # no [w] needed as word is already in syn
        if type(syn) == str:
            for t in syn.split(','): 
                t = t.replace('_', ' ')
                synset_mapping[t].append(w)
    #assert len(set(word_set)) == len(word_types)
    return synset_mapping

#########################################################################################################################################
###################################### HELPERS FOR SENTENCE / CONTEXT SAMPLING (FOR INDIVIDUAL WORD SENSES) #############################
#########################################################################################################################################


def mc_sampling(sents: list, n: int, replace=True) -> list:
    sample_inds = np.random.choice(np.arange(len(sents)), size=n, replace=replace)
    return np.array([sents[i] for i in sample_inds])


def sample_sentences(
    embeddings: dict,
    n_contexts: int,
    replace: bool
) -> Tuple[Dict[str, Dict[str, list]], List[str]]:
    samples = mc_sampling(embeddings, n_contexts, replace) 
    return samples

####################################################################################################
############################## HELPERS FOR EMBEDDING EVALUATION & SIM COMP #########################
####################################################################################################


def center_features(F: np.ndarray) -> np.ndarray:
    mu = F.mean(axis=0)
    F -= mu
    return F


def normalize_reps(F: np.ndarray) -> np.ndarray:
    F /= np.linalg.norm(F, ord=2, axis=1)[:, np.newaxis]
    return F


def find_word_id(w2v: dict, w_i: str) -> str:
    for w_j in w2v.keys():
        if re.search(f'^{w_i}', w_j):
            return w_j


def remove_rare_words(PATH: str, w2v: dict, n_contexts: int, db=None) -> dict:
    sent_counts = unpickle_file(pjoin(PATH, 'sent_counts.txt'))
    for w, count in sent_counts.items():
        if count < n_contexts:
            try:
                del w2v[w]
            except KeyError:
                if isinstance(db, pd.DataFrame):
                    try:
                        del w2v[db[db['Word'] == w]['uniqueID'].item()]
                    except (KeyError, ValueError):
                        del w2v[find_word_id(w2v, w)]
    return w2v


def get_intersection(PATH: str, n_contexts=None, sensevecs=None) -> Tuple[np.ndarray]:
    w2v = load_reps(pjoin(PATH, 'decontextualized_embeddings.txt'))
    db = get_things_db('things')
    if n_contexts:
        w2v = remove_rare_words(w2v, n_contexts, db)
    if isinstance(sensevecs, pd.DataFrame):
        db = db.iloc[sensevecs.dropna().index]
    E = []
    intersection = []
    for w, v in w2v.items():
        try:
            intersection.append(db[db['uniqueID'] == w].index[0])
        except IndexError:
            try:
                intersection.append(db[db['Word'] == w].index[0])
            except:
                continue
        E.append(v)
    return np.asarray(E), np.asarray(intersection)


def corr_mat(W: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
    W_c = W - W.mean(axis=1)[:, np.newaxis]
    cov = W_c @ W_c.T
    l2_norms = np.linalg.norm(W_c, axis=1)  # compute l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom.astype(cov.dtype)).clip(min=a_min,
                                                    max=a_max)  # counteract potential rounding errors
    return corr_mat


def fill_diag(rsm: np.ndarray) -> np.ndarray:
    """fill main diagonal of the RSM with 1"""
    assert np.allclose(
        rsm, rsm.T), '\nRSM is required to be a symmetric matrix\n'
    rsm[np.eye(len(rsm)) == 1.] = 1
    return rsm


@njit(parallel=True, fastmath=True)
def rsm_pred(W: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to representations wrt to human behavior into a RSM"""
    N = W.shape[0]
    S = W @ W.T
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    for i in prange(N):
        for j in prange(i + 1, N):
            for k in prange(N):
                if (k != i and k != j):
                    rsm[i, j] += (S_e[i, j] / (S_e[i, j] +
                                  S_e[i, k] + S_e[j, k]))
    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    return rsm


def compute_trils(W_mod1: np.ndarray, W_mod2: np.ndarray) -> float:
    assert min(W_mod1.shape) < min(
        W_mod2.shape), '\nFor RSM computation, first weight matrix must correspond to behavioral representations.\n'
    rsm_1 = fill_diag(rsm_pred(W_mod1))
    rsm_2 = corr_mat(W_mod2)
    assert rsm_1.shape == rsm_2.shape, '\nRSMs must be of equal size.\n'
    # since RSMs are symmetric matrices, we only need to compare their lower triangular parts (main diagonal can be omitted)
    tril_inds = np.tril_indices(len(rsm_1), k=-1)
    tril_1 = rsm_1[tril_inds]
    tril_2 = rsm_2[tril_inds]
    return tril_1, tril_2, tril_inds


def pearsonr(u: np.ndarray, v: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho
