import pandas as pd 
import scipy.io
from os.path import join as pjoin 
import os 


DATA_DIR = os.getenv('DATA_DIR')

def load_sorting():
    print(DATA_DIR)
    sorting_df = pd.read_csv(pjoin(DATA_DIR, 'things/unique_id.txt'), header=None, names=['concept_id'])
    return sorting_df

def load_things_database():
    sorting_df = pd.read_csv(pjoin(DATA_DIR, 'things/things_concepts.tsv'), sep='\t')
    return sorting_df

def load_behav():
    behv_sim = scipy.io.loadmat(pjoin(DATA_DIR, 'things/spose_similarity.mat'))['spose_sim']
    return behv_sim

def load_spose_similarity():
    behav_sim = load_behav()
    sorting_df = load_sorting()
    things_df = pd.DataFrame(behav_sim, index=sorting_df['concept_id'], columns=sorting_df['concept_id'])
    return things_df

def match_behv_sim(behv_sim, concepts_to_keep, sorting_df):
    concept_positions_to_keep = [sorting_df.index[sorting_df['concept_id'] == concept].tolist()[0] for concept in concepts_to_keep]
    concept_positions_to_keep = sorted(concept_positions_to_keep)
    behv_sim_matched = behv_sim[concept_positions_to_keep, :]
    behv_sim_matched = behv_sim_matched[:, concept_positions_to_keep]
    return behv_sim_matched

def load_spose_dimensions():
    rows = []
    with open(pjoin(DATA_DIR, 'things/spose_embedding_49d_sorted.txt'), 'r') as spose_file:
        for line in spose_file.read().splitlines():
            dimension_embedding = [float(value) for value in line.split(' ') if len(value)>1]
            rows.append(dimension_embedding)

    concept_ids = load_sorting()['concept_id']
    df = pd.DataFrame(rows, index=concept_ids)
    return df
    # TODO sort col index df.sort 

def load_embedding_to_df(embedding_path, matching):
    embeddings = open(embedding_path).read().splitlines()
    try:
        embeddings = [[embedding.split(';')[0], int(embedding.split(';')[1])] + [float(value) for value in embedding.split(';')[2].split(' ')] for embedding in embeddings]
    except:
        embeddings = [[embedding.split(',')[0], int(embedding.split(',')[1])] + [float(value) for value in embedding.split(',')[2].split(' ')] for embedding in embeddings]

    df = pd.DataFrame(embeddings)
    df.rename(columns={ df.columns[0]: matching, df.columns[1]: 'n_contexts'}, inplace = True)
    return df

def yield_static_data(folder):
    dir = f'{DATA_DIR}/embeddings/data/{folder}/static'
    data = (('w2v', pjoin(dir, 'w2v', 'word2vec-google-news-300', 'embeddings.txt'), 'word'),
            ('glove', pjoin(dir, 'glove', 'glove-wiki-gigaword-300', 'embeddings.txt'), 'word'),
            ('deconf', pjoin(dir, 'deconf', 'embeddings.txt'), 'synset'))

    for embedding in data:
        print(embedding)
        yield embedding

def load_things_mapping():
    path = os.path.join(DATA_DIR, 'things/things_syns.csv')
    syn_df = pd.read_csv(path)
    return syn_df

def load_sim_csv(path, sep):
    DATA_DIR = os.environ.get('DATA_DIR')
    df = pd.read_csv(os.path.join(DATA_DIR, path), sep=sep)
    return df

def load_simlex():
    path = 'SimLex-999/SimLex-999.txt'
    df = load_sim_csv(path, '\t')
    df = df.rename(columns={'SimLex999': 'human_pred'})
    return df[['word1','word2', 'human_pred']]

def load_wordsim():
    path = 'Wordsim/combined.csv'
    df = load_sim_csv(path, ',')
    df = df.rename(columns={'Human (mean)': 'human_pred', 'Word 1': 'word1', 'Word 2': 'word2'})
    return df

def get_things_id_homonym():
    things_df = load_things_database()
    ids = things_df['uniqueID']
    homonyms = [things_id for things_id in ids if any(char.isdigit() for char in things_id)]
    return homonyms

