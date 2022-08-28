from collections import defaultdict
import DecontextEmbeddings.scripts.transformer_embeddings.utils as utils
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

from os.path import join as pjoin
from copy import deepcopy 

def postprocess_embeddings(embeddings, zero_mean, unit_length, pca):
    unpacked_embeddings = []
    i_to_feature = {}
    start = 0
    for key in embeddings:
        number_embeddings = len(embeddings[key])
        for m in range(start, number_embeddings+start):
            i_to_feature[m] = key
        start = number_embeddings+start
        unpacked_embeddings += embeddings[key]
    unpacked_embeddings = np.vstack(unpacked_embeddings)

    embeddingt_dict = defaultdict(list)
    for i, embedding in enumerate(unpacked_embeddings):
        embeddingt_dict[i_to_feature[i]].append(embedding)
    
    embeddingt_stacked = {}
    for key in embeddingt_dict:
        embeddingt_stacked[key] = np.vstack(embeddingt_dict[key])
    
    return embeddingt_stacked

def decontextualize_embeddings(
    all_embeddings,
    pooling: str,
    out_path: str,
    rnd_seed: int,
    ratio=None,
    n_contexts=None,
    unit_length=False,
    all_but_the_top=False,
    output_file_name=''
) -> np.ndarray:
    """Decontextualize word sense representations across N contexts."""
    keys = []
    decontext_embeddings = []
    number_embeddings_per_key = []
    
    for (key, embeddings_of_key) in all_embeddings.items():
        actual_amount_embeddings = len(embeddings_of_key)
        
        if unit_length:
            embeddings_of_key = embeddings_of_key / np.linalg.norm(embeddings_of_key, axis=1)[:,None]

        if n_contexts != 'all':
            n_contexts = int(n_contexts)
            replace = True
            if actual_amount_embeddings > n_contexts:
                replace = False

            embeddings_of_key = utils.sample_sentences(embeddings_of_key, n_contexts, replace)
            amount_embeddings = len(embeddings_of_key)
            assert amount_embeddings == n_contexts

        decontextualized_rep = utils.aggregate_reps(embeddings_of_key, pooling, rnd_seed, ratio)
        keys.append(key)
        decontext_embeddings.append(decontextualized_rep)
        number_embeddings_per_key.append(actual_amount_embeddings)

    if all_but_the_top:
        decontext_embeddings = np.vstack(decontext_embeddings)
        original_embeddings = deepcopy(decontext_embeddings)

        scaler = StandardScaler(with_std=False)
        scaler.fit(decontext_embeddings)
        standardized_embeddings = scaler.transform(decontext_embeddings)

        D = int(standardized_embeddings.shape[1] / 100)
        u = PCA(n_components=D).fit(standardized_embeddings).components_ 
        # Subtract first `D` principal components
        # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
        decontext_embeddings = standardized_embeddings - (original_embeddings @ u.T @ u)  

    with open(pjoin(out_path, output_file_name), 'w') as f:
        for i, key in enumerate(keys):
            decontextualized_rep = decontext_embeddings[i]
            actual_amount_embeddings = number_embeddings_per_key[i]
            decontextualized_rep = ' '.join(list(map(str, decontextualized_rep)))
            f.write(f'{key};{actual_amount_embeddings};{decontextualized_rep}')
            f.write('\n')
