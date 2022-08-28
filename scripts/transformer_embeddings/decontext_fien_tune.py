import os 
from os.path import join as pjoin
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from DecontextEmbeddings.scripts.transformer_embeddings.decontextualize_embeddings import run


folders = {
    'thinga': ['word', 'synset', 'concept_id', 'main_word'],
    'word_sim': ['word']
}

models_and_layers = {
    'bert-large': range(25),
    'bert-base': range(13),
    'gpt-2': range(13),
}

n_contexts = ['all']
unit_length = False
all_but_the_top = False 


folders = {
    'things': ['word'],
}

models_and_layers = {
    'bert-base': [10]
}

for folder, synset_word_levels in folders.items():
    for model, layers in models_and_layers.items():
        for layer in layers:
            print(f'Start decontext for {model} {layer}')

            for file_name in ['spose_no_similar.txt', 'spose.txt', 'no_spose_no_similar.txt', 'no_spose.txt', 'extractions.txt']:
                path = f'../data_fine_tune/{folder}/wikidumps/decontext/{model}/{layer}/{file_name}'
                out_file_name = file_name.split('.')[0]
                run(path,
                    'mean',
                    42,
                    0.1,
                    n_contexts,
                    synset_word_levels,
                    all_but_the_top,
                    unit_length,
                    None,
                    f'{out_file_name}_decontext.txt'
                )

           