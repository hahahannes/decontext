#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from collections import defaultdict
import logging
import pickle
import os
from socket import timeout
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import random
import re
import threading
import json
import torch
import traceback
import DecontextEmbeddings.scripts.transformer_embeddings.utils as utils
from multiprocessing import Pool, Manager, Process

import numpy as np

from os.path import join as pjoin, isfile
from typing import Any, List, Tuple

FILE_NAME = 'extracted_sense_embeddings'


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--out_path', type=str)
    aa('--annotations_path', type=str)
    aa('--model_name', type=str, default='bert-large',
        choices=['bert-base', 'bert-large', 'distilbert', 'funnel-transformer', 'gpt-2', 'xlnet', 'sbert_bert', 'sbert_distill_roberta', 'gpt-2-medium'])
    aa('--batch_size', type=int, default=40)
    aa('--pooling', type=str, default="mean",
		choices=['mean', 'max', 'pca'],
		help='context pooling strategy')
    aa('--ratio', type=float, default=None,
        choices=[.1, .2, .3, .4, .5, .6, .7, .8, .9],
        help='Number of pcs to remove dependent on the variance they explain, when pca method is set to remove_first_pcs')
    aa('--rnd_seed', type=int, default=42)
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--n_jobs', default=1, help='Amount of paralle jobs', type=int)
    args = parser.parse_args()
    return args


def run(
        args,
        out_path: str,
        device: str,
) -> None:
    logger = logging.getLogger(f'main')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'main.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    m = Manager()
    logger_queue = m.Queue(10000)
    job_queue = m.Queue(10000)
    output_queue = m.Queue(10000)
	
    # Logger thread 
    lp = threading.Thread(target=utils.logger_thread, args=(logger_queue,))
    lp.start()

    #qp = threading.Thread(target=queue_watcher, args=(job_queue, logger_queue, output_queue))
    #qp.start()

    # output writer process
    # number of layers = hidden layers + input layer
    n_layers = utils.load_model(args.model_name, device).config.num_hidden_layers + 1 
    op = Process(target=write_results_to_output, args=(output_queue, out_path, n_layers))
    op.daemon = True
    op.start()
    
    # job processes
    jobs = []
    for i in range(args.n_jobs):
        logger.debug(f'start job {i}')
        job = Process(target=extract_embeddings, args=(job_queue, output_queue, args.model_name, device, logger_queue, i))
        job.daemon = True
        job.start()
        jobs.append(job)

    annotations_path = args.annotations_path
    for line in utils.file_to_lines(annotations_path):
        mapping = line.split(';')[0]
        text = line.split(';')[1]
        job_queue.put((text, mapping))

    logger.debug(f'Number of remaining jobs after creating all jobs: {job_queue.qsize()}')	
                
    logger.debug('put termination signal in job queue')
    for _ in range(args.n_jobs):
        job_queue.put(None)

    logger.debug('wait for jobs')
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug('join: ' + current_time)
    logger.debug(f'Number of remaining jobs after creating all jobs: {job_queue.qsize()}')
    logger.debug(f'Number of remaining output jobs: {output_queue.qsize()}')
    for job in jobs:
        job.join()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug('after join: ' + current_time)
    logger.debug(f'Number of remaining jobs after join: {job_queue.qsize()}')

    # put termination signals in other queues after job processes are done! otherwise early stop
    logger.debug('put termination signal in output queue and wait')
    logger.debug(f'Number of remaining output jobs: {output_queue.qsize()}')
    output_queue.put(None)
    op.join()

    logger.debug('put termination signal in logger queue and wait')
    logger_queue.put(None)
    lp.join()
    logger.debug('logger job done')

    logger.debug('extraction jobs done')

if __name__ == '__main__':
    # parse arguments
    args = parseargs()
    # set random seeds
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    out_path = pjoin(args.out_path, args.model_name)

    if not os.path.exists(out_path):
        print('\nCreating directories...\n')
        os.makedirs(out_path)

    run(
        args=args,
        out_path=out_path,
        device=device,
    )
