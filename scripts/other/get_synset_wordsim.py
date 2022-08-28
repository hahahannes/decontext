import pandas as pd 
from nltk.corpus import wordnet

import os 

DATA_DIR = '/home/hhansen/decon/decon_env/DecontextEmbeddings/data'
os.environ['DATA_DIR'] = DATA_DIR

import sys
sys.path.append('/home/hhansen/decon/decon_env/DecontextEmbeddings')
from helpers.data import load_simlex, load_wordsim

simlex = load_simlex()
wordsim = load_wordsim()

words = pd.DataFrame({'word': list(set(simlex['word1']).union(set(simlex['word2'])).union(set(wordsim['word1'])).union(set(wordsim['word2'])))})

def get_syn(word):  
    syns = wordnet.synsets(word)
    if syns:
        # syns[0].name() -> bank.n.01
        return syns[0].lemmas()[0].key()

words['synset'] = words['word'].apply(get_syn)

words.to_csv('wordsim_synsets.csv', index=False)
