import os 
from os.path import join as pjoin
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from DecontextEmbeddings.scripts.transformer_embeddings.decontextualize_embeddings import run


n_contexts = [1, 10, 50, 100, 300, 500, 1000, 'all']
unit_length = False
all_but_the_top = False 

folders = {
    'thinga': ['word', 'synset', 'concept_id', 'main_word'],
    'word_sim': ['word', 'main_word']
}

models_and_layers = {
    'gpt-2': range(13),
    'bert-base': range(13),
    'bert-large': range(25),
    'gpt-2-medium': range(25),
    'sbert_bert': range(13)
}


for folder, synset_word_levels in folders.items():
    for model, layers in models_and_layers.items():
        for layer in layers:
            print(f'Start decontext for {model} {layer}')

            path = f'../../data/{folder}/wikidumps/decontext/{model}/{layer}/extractions.txt'
            run(path,
                    'mean',
                    42,
                    0.1,
                    n_contexts,
                    synset_word_levels,
                    all_but_the_top,
                    unit_length,
                    None,
                    f'decontext.txt'
            )

           
