from helpers.data import yield_static_data
from helpers.things_evaluation.evaluate import read_embeddings
from helpers.embedding_evaluation.data import read_wordsim_embeddings
import os

EMBEDDING_DATA_DIR = os.getenv('EMBEDDING_DATA_DIR')

def get_intersection_words(min_n_contexts=None, corpus_folder='wikidumps', matching='word', folder='thinga'):
    """
    Get intersection of words that have embeddings in Transformer extracted embeddings and static embeddings
    """
    
    # use one embedding based on synset to get found words -> could be less because only matching synsets a re used

    if folder == 'thinga':
        embedding_path = f'{EMBEDDING_DATA_DIR}/{folder}/{corpus_folder}/decontext/bert-base/0/{matching}/mean/1/decontext.txt'
        embedding_df = read_embeddings(embedding_path, matching=matching, min_n_contexts=min_n_contexts)
    else:
        embedding_path = f'{EMBEDDING_DATA_DIR}/{folder}/{corpus_folder}/decontext/bert-base/0/word/mean/1/decontext.txt'
        embedding_df = read_wordsim_embeddings(embedding_path, matching='word', min_n_contexts=min_n_contexts, as_df=True)

    matching_words = set(embedding_df.index)
    # remove rare features -> n context per word and per synset matchen -> unterschieldich

    for model, embedding_path, matching in yield_static_data(folder):
        if folder == 'thinga':
            embedding_df = read_embeddings(embedding_path, matching=matching)
        else:
            embedding_df = read_wordsim_embeddings(embedding_path, matching=matching, as_df=True)
        matching_words = matching_words.intersection(embedding_df.index)
    
    print(f'Number of words intersection: {len(matching_words)}')
    return matching_words