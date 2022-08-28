from collections import defaultdict
from email.policy import default
import torch

torch.set_num_threads(10)

from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time 
import random
import numpy as np
import pandas as pd 
from multiprocessing import Manager, Pool, Process
import os 
import argparse
from os.path import join as pjoin
from model import SiameseNetwork

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--path', type=str)
    aa('--n_jobs', type=int)
    #aa('--file_name', type=str)
    #aa('--checkpoint_path', type=str)
    args = parser.parse_args()
    return args

def run_inference_single_embedding(job_queue, output_queue, checkpoint_path):
    torch.set_num_threads(1)
    model = SiameseNetwork(768)
    states = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(states)
    
    model.eval()

    while True:
        record = job_queue.get()
        if record is None:
            break
        word, synset, embedding = record
        transformed_embedding1, _ = model(embedding, embedding)
        transformed_embedding1 = transformed_embedding1.detach().cpu().numpy()
        output_queue.put((word, synset, transformed_embedding1))

def write_results_to_output(output_queue, out_path, job_queue, name):
    with open(os.path.join(out_path, name), 'w') as output_file:
        counter = 0
        while True:
            record = output_queue.get()
            if record is None:
                break
            word, synset, embedding = record
            embedding_str = ' '.join(str(value) for value in embedding)
            output_file.write(';'.join([word, synset, embedding_str]))
            output_file.write('\n')

            if counter % 10000 == 0:
                print(f'Jobs left in queue {job_queue.qsize()}')
            counter += 1

def run_inference(out_path, n_jobs, embedding_path, file_name, checkpoint_path):
    m = Manager()
	
    output_queue = m.Queue()
    job_queue = m.Queue()

    op = Process(target=write_results_to_output, args=(output_queue, out_path, job_queue, file_name))
    op.start()

    jobs = []
    for i in range(n_jobs):
        job = Process(target=run_inference_single_embedding, args=(job_queue, output_queue, checkpoint_path))
        job.daemon = True  
        job.start()
        jobs.append(job)	

    with open(embedding_path, 'r') as embeddings_file:
        for line in embeddings_file:
            line = line.strip()
            split = line.split(';')
            word = split[0]
            synset = split[1]
            embedding = split[2].split(' ')
            embedding = torch.as_tensor([float(value) for value in embedding])
            job_queue.put((word, synset, embedding))		
			
    for _ in jobs:
        job_queue.put(None)

    for w in jobs:
        w.join()


    output_queue.put(None)
    op.join()


if __name__ == '__main__':
    args = parseargs()
    layers = range(13)
    #torch.set_num_threads(1)

    for file_name, checkpoint_path in [
            #('mse_loss.txt', 'mse_loss/checkpoint.pth.tar'),
            #('constrastive_loss.txt', 'contrastive_loss/checkpoint.pth.tar'),
            ('mse_new.txt', 'mse_new/checkpoint.pth.tar')

    ]:
        model = SiameseNetwork(768)
        states = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(states)
        model.eval()
        
        for layer in layers:
            print(f'{file_name} - {layer}')
            base_path = f'../../../data_fine_tune/things/wikidumps/decontext/bert-base/{str(layer)}/word/mean/all/'
            embedding_path = pjoin(base_path, 'decontext.txt')
            
            with open(os.path.join(base_path, file_name), 'w') as output_file:
                with open(embedding_path, 'r') as embeddings_file:
                    for line in embeddings_file:
                        line = line.strip()
                        split = line.split(';')
                        word = split[0]
                        n_contexts = split[1]
                        embedding = split[2].split(' ')
                        embedding = torch.as_tensor([float(value) for value in embedding])
                        transformed_embedding1, _ = model(embedding, embedding)
                        transformed_embedding1 = transformed_embedding1.detach().cpu().numpy()
    
                        embedding_str = ' '.join(str(value) for value in transformed_embedding1)
                        output_file.write(';'.join([word, n_contexts, embedding_str]))
                        output_file.write('\n')
            #run_inference(out_path, args.n_jobs, embedding_path, file_name, checkpoint_path)
