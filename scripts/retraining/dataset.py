from collections import defaultdict
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd 
import scipy.io


def load_behav():
    # Read similarity data from THINGS behaviourial data
    behv_sim = scipy.io.loadmat('../../data/things/spose_similarity.mat')['spose_sim']
    return behv_sim

def load_sorting():
    # Read things concept sorting
    sorting_df = pd.read_csv('../../data/things/unique_id.txt', header=None, names=['concept_id'])
    return sorting_df

def load_things():
    behav_sim = load_behav()
    sorting_df = load_sorting()
    things_df = pd.DataFrame(behav_sim, index=sorting_df['concept_id'], columns=sorting_df['concept_id'])
    things_df = things_df.reindex(sorted(things_df.columns), axis=1)
    things_df = things_df.reindex(sorted(things_df.columns), axis=0)
    return things_df


class EmbeddingDataSet(Dataset):
    def __init__(self,embeddings_path, word_path):
        self.embeddings_path = embeddings_path
        self.word_path = word_path
        self.embeddings = defaultdict(list)
        self.things = load_things()
        self.load_embeddings()
        self.words = list(pd.read_csv(self.word_path, header=None, names=['words'])['words'])
        
    def load_embeddings(self):
        with open(self.embeddings_path, 'r') as embeddings_file:
            for line in embeddings_file:
                line = line.strip()
                split = line.split(';')
                word = split[0]

                # word to things_id
                word = word.replace(' ', '_')
                embedding = split[2].split(' ')
                embedding = torch.as_tensor([float(value) for value in embedding])
                self.embeddings[word].append(embedding)
    
    def __len__(self):
        n = len(self.embeddings)
        return int(n * (n-1) / 2)


class ThingsDataSet(EmbeddingDataSet):
    def __init__(self,embeddings_path, word_path):
        super().__init__(embeddings_path, word_path)

    def __getitem__(self, index):
        # choose random words
        word1 = 'dog' #random.choice(self.words)
        word2 = 'table' #random.choice(self.words)

        similarity = self.things.loc[word1, word2]

        embedding1 = random.choice(self.embeddings[word1])
        embedding2 = random.choice(self.embeddings[word2])
        
        return embedding1, embedding2, torch.from_numpy(np.array([similarity], dtype=np.float32))

class FileDataSet():
    def __init__(self, dataset_path):
        self.embeddings_list = []
        with open(dataset_path, 'r') as embedding_file:
            for line in embedding_file.readlines():
                embedding1, embedding2, similarity = line.split(';')
                embedding1 = torch.as_tensor([float(value) for value in embedding1.split(' ')])
                embedding2 = torch.as_tensor([float(value) for value in embedding2.split(' ')])
                similarity = torch.from_numpy(np.array([similarity], dtype=np.float32))
                self.embeddings_list.append((embedding1, embedding2, similarity))

    def __getitem__(self, index):
        return self.embeddings_list[index]

    def __len__(self):
        return len(self.embeddings_list)

# instead of spose ground truth
class WordpairDataSet(EmbeddingDataSet):
    def __init__(self,embeddings_path, word_path):
        super().__init__(embeddings_path, word_path)

    def __getitem__(self, index):
        should_get_same_class = random.randint(0,1) 

        if should_get_same_class:
            word = random.choice(self.words)
            similarity = 1
            embedding1 = random.choice(self.embeddings[word])
            embedding2 = random.choice(self.embeddings[word])
        else:
            word1 = random.choice(self.words)
            word2 = random.choice(self.words)

            similarity = 0

            embedding1 = random.choice(self.embeddings[word1])
            embedding2 = random.choice(self.embeddings[word2])

        return embedding1, embedding2, torch.from_numpy(np.array([similarity], dtype=np.float32))

    def __len__(self):
        return 1000