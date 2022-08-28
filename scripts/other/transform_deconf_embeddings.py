import pandas as pd 
import os 

EMBEDDING_DATA_DIR = '/home/hhansen/decon/decon_env/data'
os.environ['EMBEDDING_DATA_DIR'] = EMBEDDING_DATA_DIR
os.environ['EMBEDDING_EVALUATION_DATA_PATH'] = '/home/hhansen/decon/decon_env/DecontextEmbeddings/helpers/embedding_evaluation/data/'
DATA_DIR = '/home/hhansen/decon/decon_env/DecontextEmbeddings/data'
os.environ['DATA_DIR'] = DATA_DIR

from data import load_things_database

deconf_embd = pd.read_csv(f'{EMBEDDING_DATA_DIR}/thinga/static/deconf/deconf.csv', header=None)
things_df = load_things_database()
print(things_df)

with open(f'{EMBEDDING_DATA_DIR}/thinga/static/deconf/synset.csv', 'w') as o_file:
    for i, synset_id in enumerate(list(things_df['Wordnet ID2'])):
        if not pd.isna(synset_id):
            o_file.write(synset_id)
            o_file.write(',0,')
            o_file.write(' '.join(map(str, deconf_embd.iloc[i, :])))
            o_file.write('\n')

with open(f'{EMBEDDING_DATA_DIR}/thinga/static/deconf/concept_id.csv', 'w') as o_file:
    for i, synset_id in enumerate(list(things_df['Wordnet ID4'])):
        if not pd.isna(synset_id):
            o_file.write(synset_id)
            o_file.write(',0,')
            o_file.write(' '.join(map(str, deconf_embd.iloc[i, :])))
            o_file.write('\n')