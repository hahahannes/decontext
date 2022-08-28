#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from nltk.corpus import wordnet 

def get_synset_id(key):
    for ss in wordnet.synsets(key.split('%')[0]):
        for lemma in ss.lemmas():
            if lemma.key() == key:
                return ss.name()

def lemma_ares_mapping(ares_path, wordset, syns={}, sep=" "):
    # car - automobile have own ares synsets 
    # ticktacktoe has no synset - tic-tac-toe has 

    # TODO only take similary synsets 
    embedding_mapping = defaultdict(dict)
    ares_mapping = defaultdict(dict)

    with open(ares_path, "r") as f:
        for line in f:
            line = line.strip().split(sep)
            key = line[0]
            lemma = key.split('%')[0]
            lemma = lemma.replace('_', ' ')
            rep = np.array(list(map(float, line[1:])))   
            syn_id = get_synset_id(key) 
            ares_mapping[lemma][key] = (rep, syn_id) 
            
    # use synset embedding of a specfic word even if synonyms
    # car -> car%1.00 automobile -> automobile%1.00  
    # only use other synset embedding if word does not have one present in ARES
    for word in wordset:
        word = word.replace('_', ' ')
        if word in ares_mapping:
            embedding_mapping[word] = ares_mapping[word]
        else:
            # search for synonyms that are in ARES
            # eg. ticktacktoe has no ARES embedding but synonym tic-tac-toe has
            for syn, synonym_words in syns.items():
                if word in synonym_words and syn in ares_mapping:
                    embedding_mapping[word] = ares_mapping[syn]

    for synonym, synonym_words in syns.items():
        synonym = synonym.replace('_', ' ')
        if synonym in ares_mapping:
            embedding_mapping[synonym] = ares_mapping[synonym]
        else:
            # search for words of synonym that are in ARES
            for word in syns[synonym]:
                if word in ares_mapping:
                    embedding_mapping[synonym] = ares_mapping[word]
    
    # synonyms that have no key 
    #for syn in syns:
    #    if 
    return embedding_mapping
