#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
import pickle
import os
import random
import torch
import DecontextEmbeddings.scripts.transformer_embeddings.utils as utils
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

from os.path import join as pjoin, split
from typing import Any, List, Tuple
from copy import deepcopy 

torch.set_num_threads(1)


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--embeddings_path', type=str)
    aa('--synset_word_level', type=str)
    aa('--all_but_the_top', action='store_true')
    aa('--unit_length', action='store_true')
    aa('--pooling', type=str,
        choices=['mean', 'max', 'pca', 'regression'],
        default='mean',
        help='context pooling strategy')
    aa('--n_contexts', type=str,
        help='number of contexts to sample per word sense')
    aa('--ratio', type=float, default=None,
        choices=[.1, .2, .3, .4, .5, .6, .7, .8, .9],
        help='Number of pcs to remove dependent on the variance they explain, when pca method is set to remove_first_pcs')
    aa('--rnd_seed', type=int, default=42)
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--max_number_of_embeddings', type=int)
    aa('--output_file_name', type=str, default='decontextualized_sense_embeddings.txt')
    args = parser.parse_args()
    return args


def run(
        embeddings_path,
        pooling,
        rnd_seed,
        ratio,
        n_contexts,
        synset_word_levels,
        all_but_the_top=False,
        unit_length=False,
        max_number_of_embeddings=None,
        output_file_name='decontext.txt'
) -> None:
    for synset_word_level in synset_word_levels:
        print(f'level: {synset_word_level}')
        embeddings = utils.load_embeddings(embeddings_path, synset_word_level, max_number_of_embeddings)
        for n_context in n_contexts:
            splitted_path = list(split(embeddings_path))
            aggregation_str = f'{pooling}'
            if all_but_the_top:
                aggregation_str += '_abtt'
            if unit_length:
                aggregation_str += '_unit_length'

            out_path = splitted_path[:-1] + [synset_word_level, aggregation_str, str(n_context)] 
            out_path = pjoin(*out_path)
            print(out_path)

            if not os.path.exists(out_path):
                print(f'\nCreating directories {out_path}\n')
                os.makedirs(out_path)

                # decontextualize representations
                print('start decontext')
                decontextualize_embeddings(
                        all_embeddings=embeddings,
                        pooling=pooling,
                        out_path=out_path,
                        rnd_seed=rnd_seed,
                        ratio=ratio,
                        n_contexts=n_context,
                        unit_length=unit_length,
                        all_but_the_top=all_but_the_top,
                        output_file_name=output_file_name
                )


if __name__ == '__main__':
    # parse arguments
    args = parseargs()
    # set random seeds
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    
    run(
        args.embeddings_path,
        args.pooling,
        args.rnd_seed,
        args.ratio,
        args.n_contexts.split(','),
        args.synset_word_level.split(','),
        args.all_but_the_top,
        args.unit_length,
        args.max_number_of_embeddings,
        args.output_file_name
    )
