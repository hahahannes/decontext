import os

from load_embeddings import get_intersection_words

out_path = '/home/hhansen/.cognival/embeddings'

layers = range(13)
n_contexts = ['1', 'all']
models = ['bert-base', 'gpt-2']

matching = get_intersection_words(None, 'wikidumps', 'word', 'cognival')
paths = []

for layer in layers:
    for model in models:
        for n_context in n_contexts:
            embedding_path = f'../../data/cognival/wikidumps/decontext/{model}/{layer}/word/mean/{n_context}/decontext.txt'
            if model == 'bert-base':
                dir_name = 'bert.base.'
            else:
                dir_name = 'gpt2.'
            dir_name += str(layer) + '.' + n_context
            paths.append((embedding_path, dir_name))

paths = []
paths.append(('../../data/cognival/static/w2v/word2vec-google-news-300/embeddings.txt', 'own_word2vec'))
paths.append(('../../data/cognival/static/glove/glove-wiki-gigaword-300/embeddings.txt', 'own_glove300'))

for path, dir_name in paths:
    embeddings = open(path).read().splitlines()

    out_path_dir = os.path.join(out_path, dir_name)
    if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)

    with open(os.path.join(out_path_dir, 'cognival.txt'), 'w') as f:
        for embedding in embeddings:
            word = embedding.split(';')[0]    
            if word in matching:            
                decontextualized_rep = ' '.join(list(map(str, [float(value) for value in embedding.split(';')[2].split(' ')] )))
                f.write(f'{word} {decontextualized_rep}')
                f.write('\n')