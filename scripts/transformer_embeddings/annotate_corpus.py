#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from email.mime import base
import pickle
import DecontextEmbeddings.scripts.transformer_embeddings.utils as utils
import time
from collections import defaultdict
from multiprocessing import Manager, Pool, Process
from unidecode import unidecode
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import logging
import traceback
import re 
import json 
import threading
import traceback
from os.path import join as pjoin, isfile

import numpy as np
import torch
from DecontextEmbeddings.scripts.transformer_embeddings.utils import mc_sampling

from data.corpus_sampling.corpus_utils import word_to_corp_ind, find_texts
from data.corpus_sampling.ares_utils import lemma_ares_mapping

import argparse

def parseargs():
	parser = argparse.ArgumentParser()
	def aa(*args, **kwargs):
		parser.add_argument(*args, **kwargs)
	aa('--corpus_path', type=str)
	aa('--out_path', type=str)
	aa('--wordset_path', type=str)
	aa('--wordset_format', type=str,
		choices=['things', 'simlex999'],
		help='Format of words to build embeddings for')
	aa('--model_name', type=str, default="bert-large",
		choices=['bert-base', 'bert-large', 'distilbert', 'funnel-transformer', 'gpt-2', 'xlnet'])
	aa('--n_layers', type=int, default=4,
		help='number of Transformer layers over which to aggregate')
	aa('--batch_size', type=int, default=1,
		help='batch size for processing sentences')
	aa('--aggregation', type=str, default="top",
		choices=['bottom', 'top', 'intermediate'],
		help='whether to perform aggregation across bottom, top, or intermediate l layers')
	aa('--pooling', type=str, default="mean",
		choices=['mean', 'max', 'pca'],
		help='context pooling strategy')
	aa('--grouping', type=str, default="ares",
		choices=['ares', 'unsupervised','none'],
		help='grouping method')
	aa('--ares_path', type=str)
	aa('--wikidumps', action='store_true')
	aa('--annotate', action='store_true')
	aa('--device', type=str, default='cuda',
		choices=['cpu', 'cuda'])
	aa('--n_jobs', default=1, help='Amount of paralle jobs', type=int)
	aa('--upper_limit', default=100, help='Upper limit of contexts', type=int)
	args = parser.parse_args()
	return args


def ares_grouping_method(embeddings, ares_reps):
	closest_reps = []
	for embedding in embeddings:
		stacked_word_rep = np.concatenate([embedding, embedding])
		closest_reps.append(utils.nearest_neighbor(stacked_word_rep, ares_reps))
	return closest_reps

def write_results_to_output(output_queue, logger_queue, out_path):
	with open(pjoin(out_path, 'annotations.txt'), 'w') as output_file:
		while True:
			record = output_queue.get()
			if record is None:
				break
			text, mapping = record
			text = text.replace('\n', '')
			
			output_file.write(';'.join([json.dumps(mapping), text]))
			output_file.write('\n')

def run(job_queue, output_queue, model_name, device, logger_queue, i):
	logger = logging.getLogger(f'{i}')
	logger.setLevel(logging.DEBUG)
	# create file handler which logs even debug messages
	fh = logging.FileHandler(f'{i}.log')
	fh.setLevel(logging.DEBUG)
	logger.addHandler(fh)

	base_model = utils.load_model(model_name, device)
	tokenizer = utils.load_tokenizer(model_name)

	rep = None
	rematched_word = None
	text = None

	while True:
		record = job_queue.get()
		if record is None:
			break
		try:
			text, mapping = record
			logger.debug(f'Text: {text}')
			logger.debug(f'Found words: {mapping}')

			results = []

			torch.set_num_threads(1)
			ares_mapping = None
			strip_first_last_token = False
			if re.search(r'.*bert', model_name):
				strip_first_last_token = True
			
			with open('ares', 'rb') as file:
				ares_mapping = pickle.load(file)

			processed = utils.tokenize_sentence(tokenizer, text, True, True, False).to(base_model.device)
			number_of_tokens = len(processed['input_ids'][0])
			if number_of_tokens > tokenizer.model_max_length:
				continue 
			
			all_token_embeddings = utils.get_hiddens(base_model, processed)
			offset_mappings = processed['offset_mapping']

			for word_found_in_sentence, main_words in mapping:		
				logger.debug(f'get embeddings for word: {word_found_in_sentence} - main words: {main_words}')			
				embeddings, token_ranges = utils.get_embeddings_of_word(word_found_in_sentence, processed, all_token_embeddings, tokenizer, strip_first_last_token=strip_first_last_token)
				# mean pool last 4 layers according to ARES
				embeddings = [embedding[21:].mean(dim=0) for embedding in embeddings]

				#TODO: implement for unsupervised case
				if args.grouping == "ares":
					if ares_mapping[word_found_in_sentence]:
						groups = ares_grouping_method(embeddings, ares_mapping[word_found_in_sentence])
					else:
						groups = [('no_ares_rep', 'no_ares_rep') for i in embeddings]
									
					for i, rep in enumerate(groups):
						#sent_number = sen_inds[sen_ind]
						offset_mapping_of_sentence = offset_mappings[0]
						start = token_ranges[i][0]
						end = token_ranges[i][1]
						string_start = offset_mapping_of_sentence[start][0].item()
						string_end = offset_mapping_of_sentence[end][1].item()

						# problem pédal - pedal
						rematched_word = text[string_start:string_end]
						rematched_word = rematched_word.lower().strip()
						rematched_word = unidecode(rematched_word)
						assert rematched_word == word_found_in_sentence.lower().strip()

				result = (word_found_in_sentence, main_words, groups)
				results.append(result)

			output_queue.put((text, results))
					# TODO add synset ids eg bike.n.02 not only word synset id bike%2:80 maybe in ares_utils

		except Exception as e:
			e = traceback.print_exception(value=e, tb=e.__traceback__, etype=Exception)
			logger_queue.put(f'Exc: {e} in job {i}')

def sample_contexts(upper_limit, occurences_path, limited_occurences_path, logger):
	logger.debug('SAMPLE: Count all occurences')
	word_to_text = defaultdict(list)
	for line in utils.file_to_lines(occurences_path):
		text_id = line.split(';')[0]
		words_found = line.split(';')[1]
		words_found = json.loads(words_found)
		main_words = []
		for word_mapping in words_found:
			main_words += word_mapping[1]
		
		for main_word in main_words:
			word_to_text[main_word].append(text_id)
		
	logger.debug(f'SAMPLE: sample maximum {upper_limit}')
	word_to_text_with_limit = defaultdict(list)
	for word_found, text_ids in word_to_text.items():
		number_contexts = len(text_ids)
		if number_contexts > upper_limit:
			sampled_text_ids = mc_sampling(text_ids, upper_limit, replace=False)
			word_to_text_with_limit[word_found] = sampled_text_ids
		else:
			word_to_text_with_limit[word_found] = text_ids

	logger.debug(f'SAMPLE: Write samples')
	with open(limited_occurences_path, 'w') as output_file:
		for line in utils.file_to_lines(occurences_path):
			splitted = line.split(';')
			text_id = splitted[0]
			text = splitted[2].replace('\n', '')
			
			all_words_found = splitted[1]
			all_words_found = json.loads(all_words_found)
			words_found_in_text_and_sampled = []

			for word_mapping in all_words_found:
				word_belongs_to_sampled_text = False
				main_words = word_mapping[1]
				for main_word in main_words:
					if text_id in word_to_text_with_limit[main_word]:
						word_belongs_to_sampled_text = True
				if word_belongs_to_sampled_text:
					words_found_in_text_and_sampled.append(word_mapping)
				
			if words_found_in_text_and_sampled:
				output_file.write(';'.join([json.dumps(words_found_in_text_and_sampled), text]))
				output_file.write('\n')
	
if __name__ == '__main__':
	start_time = time.time()
	logger = logging.getLogger(f'main')
	logger.setLevel(logging.DEBUG)
	# create file handler which logs even debug messages
	fh = logging.FileHandler(f'main.log')
	fh.setLevel(logging.DEBUG)
	logger.addHandler(fh)

	args = parseargs()
	path_to_corpus = args.corpus_path
	path_to_wordset = args.wordset_path
	out_path = args.out_path

	model_name = args.model_name
	n_layers = args.n_layers
	aggregation = args.aggregation
	pooling = args.pooling
	batch_size = args.batch_size
	device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

	logger.debug('Synonym mapping')
	if args.wordset_format in ["things", "simlex999"]:
		word_set = utils.csv_to_list(path_to_wordset, ',', "Word")
		synsets = utils.csv_to_list(path_to_wordset, ',', "Wordnet Synset")
		synset_mapping = utils.syns_to_words(word_set, synsets)
	else:
		word_set = open(path_to_wordset, 'r').read().split('\n')
		synset_mapping = dict()
	logger.debug(synset_mapping)

	occurences_path = pjoin(out_path, 'all_occurences.txt')
	limited_occurences_path =  pjoin(out_path, 'limited_occurences.txt')
	
	if not os.path.isfile(occurences_path):
		# find text where words/synonyms occur -> save in file
		logger.debug('FIND: start findind text where words occur')
		find_texts(path_to_corpus, word_set, synset_mapping, logger, occurences_path, args.n_jobs, args.wikidumps)

	if not os.path.isfile(limited_occurences_path):
		# sample upper limit 
		logger.debug('SAMPLE: start sampling text where words occur with upper limit')
		sample_contexts(args.upper_limit, occurences_path, limited_occurences_path, logger)

	if args.annotate:		
		with torch.no_grad():
			if args.grouping == "ares":
				logger.debug('find ares mappings')
				ares_mapping = lemma_ares_mapping(args.ares_path, word_set, synset_mapping)
				#print(list(ares_mapping['car'].keys()))
				with open('ares', 'wb') as file:
					file.write(pickle.dumps(ares_mapping))

			# dont use error callback no info where exception occured
			#def callback(e):
			#	logger.debug(e)

			logger.debug('Annotate sentences')
			m = Manager()
			logger_queue = m.Queue()
			job_queue = m.Queue()
			output_queue = m.Queue()
			lp = threading.Thread(target=utils.logger_thread, args=(logger_queue,))
			lp.start()

			op = Process(target=write_results_to_output, args=(output_queue, logger_queue, out_path))
			op.start()

			jobs = []
			for i in range(args.n_jobs):
				logger.debug(f'start job {i}')
				job = Process(target=run, args=(job_queue, output_queue, model_name, device, logger_queue, i))
				job.daemon = True  
				job.start()
				jobs.append(job)	

			for line in utils.file_to_lines(limited_occurences_path):
				mapping = json.loads(line.split(';')[0])
				text = line.split(';')[1]
				job_queue.put((text, mapping))		
			
			for _ in jobs:
				job_queue.put(None)

			for w in jobs:
				w.join()

			logger.debug('jobs done')
			logger_queue.put(None)
			lp.join()

			output_queue.put(None)
			op.join()

		end_time = time.time()
		logger.debug(f'Annotation took {end_time-start_time}')
