# static embeddings 
import gensim.downloader
import pickle
import os 
from os.path import join as pjoin
import pandas as pd
#import openai

DATA_DIR = '/home/hhansen/decon/decon_env/data'
EMBEDDING_DATA_DIR = '/home/hhansen/decon/decon_env/DecontextEmbeddings/data'

def load_things():
    return pd.read_csv(f'{EMBEDDING_DATA_DIR}/things/item_names.tsv', sep='\t')

def get_embedding(model, word_set):
    embeddings = {}
    for key in word_set:
        print(f'Get word2vec embedding for {key}')
        try:
            embeddings[key] = model[key]
        except KeyError:
            # word not found in w2v
            continue
    return embeddings

def get_word2vec_embeddings(word_set, pretrained_model):
    print('load word2vec model')
    w2v = gensim.downloader.load(pretrained_model)
    return get_embedding(w2v, word_set)

def get_glove_embeddings(word_set, pretrained_model):
    print('load glove model')
    glove = gensim.downloader.load(pretrained_model)
    return get_embedding(glove, word_set)

def get_deconf(word_set):
    mapping_df = pd.read_csv(f'{EMBEDDING_DATA_DIR}/deconflated/mapping.txt', sep='\t', header=None, names=['id', 'synset_id'])
    embeddings_df = pd.read_csv(f'{EMBEDDING_DATA_DIR}/deconflated/embeddings.txt', sep=' ', header=None)
    embeddings_df = embeddings_df.rename(columns={embeddings_df.columns[0]: 'id'})
    embeddings_df = embeddings_df.merge(mapping_df, on='id').drop(columns=['id']).set_index('synset_id')
    
    embeddings = {}
    for row in embeddings_df.iterrows():
        key = row[0]
        embeddings[key] = list(row[1])

    return get_embedding(embeddings, word_set)

def get_gpt3_embedding(word_set, pretrained_model):
    print(pretrained_model)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings = {}
    for row in word_set.itertuples():
        word = row.Word
        print(f'Get OpenAI embedding for {word}')
        try:
            word = word.replace('\n', ' ')
            embedding = openai.Engine(id=pretrained_model).embeddings(input = [word])['data'][0]['embedding']
        except KeyError:
            # word not found in w2v
            continue
        embeddings[word] = embedding
    return embeddings


if __name__ == '__main__':
    word_set = 'things'
    
    embedding_types = ['deconf']
    pretrained_models = [(0)]

    use_synset_id = True

    #embedding_types = ['w2v', 'glove']
    #pretrained_models = [
    #    ('word2vec-google-news-300',),
    #    ('glove-wiki-gigaword-300',)
    #]

    word_list = []
    embeddings = None

    if word_set == 'things':
        df = load_things()[['Word', 'Wordnet ID2']]
        if use_synset_id:
            word_list = df['Wordnet ID2']
        else:
            word_list = df['Word']

    if word_set == 'word_sim':
        if use_synset_id:
            df = pd.read_csv(f'{DATA_DIR}/word_sim/wordsim_synsets.csv')
            word_list = df['synset']
        else:
            df = pd.read_csv(f'{DATA_DIR}/word_sim/word_sim_vocabulary.csv', header=None, names=['Word'])
            word_list = df['Word']

    if word_set == 'cognival':
        word_list = pd.read_csv(f'{DATA_DIR}/cognival/vocab.txt', header=None, names=['Word'])

    for i, embedding_type in enumerate(embedding_types):
        if embedding_type == 'w2v':
            embeddings = get_word2vec_embeddings(word_list, pretrained_model)
        elif embedding_type == 'glove':
            embeddings = get_glove_embeddings(word_list, pretrained_model)
        elif embedding_type == 'gpt-3':
            embeddings = get_gpt3_embedding(word_list, pretrained_model)
        elif embedding_type == 'deconf':
            embeddings = get_deconf(word_list)

        out_path = f'{DATA_DIR}/{word_set}/static'
        out_path = pjoin(out_path, embedding_type)

        if not os.path.exists(out_path):
            print('\nCreating directories...\n')
            os.makedirs(out_path)

        with open(pjoin(out_path, 'embeddings.txt'), 'w') as f:
            for word in embeddings:
                f.write(f'{word};0;')
                f.write(' '.join(str(value) for value in embeddings[word]))
                f.write('\n')