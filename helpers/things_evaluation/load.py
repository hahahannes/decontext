from load_embeddings import yield_decon_data, yield_static_data
from things_evaluation.evaluate import read_embeddings
import os 
from os.path import join as pjoin

def get_intersection_words_things(min_n_contexts=None, corpus_folder='wikidumps', matching='synset'):
    matching_words = set()
    folder = 'thinga'
    # number of words should be same for all decontext embeddings -> use one setting
    # use one embedding absed on words to get found words
    #embedding_path = '../../data/thinga/output/decontext/bert-base/00/00/True/word/mean/1/decontextualized_sense_embeddings.txt'
    #embedding_df = read_embeddings(embedding_path, 'word', matching_words=None, min_n_contexts=min_n_contexts)
    #matching_words = set(embedding_df.index)
    #print(f'{embedding_path} {len(matching_words)}')

    # use one embedding based on synset to get found words -> could be less because only matching synsets a re used
    embedding_path = pjoin(os.environ.get('DATA_DIR'), f'embeddings/data/thinga/{corpus_folder}/decontext/bert-base/0/{matching}/mean/1/decontext.txt')
    embedding_df = read_embeddings(embedding_path, matching, matching_words=None, min_n_contexts=min_n_contexts)
    matching_words = set(embedding_df.index)
    print(f'{embedding_path} {len(matching_words)}')
    # remove rare features -> n context per word and per synset matchen -> unterschieldich

    for model, detail_model, embedding_path in yield_static_data(folder):
        matching = 'word'
        embedding_df = read_embeddings(embedding_path, matching, matching_words=None)
        matching_words = matching_words.intersection(embedding_df.index)
        print(f'{model} {len(matching_words)}')


    return matching_words