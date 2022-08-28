#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import inflect
from nltk.corpus import wordnet as wn
import nltk
nltk.download('punkt')
import numpy as np 
import logging 
from multiprocessing import Pool, Manager, Process
import sys 
import json 
import functools
from os.path import join as pjoin, isfile
import utils
import threading
import itertools as IT
import os 
import re

# TODO word im haben pos -> zu wordnet umwaqndle
# TODO extract pos from things synset 
# all here not in file 


def find_text(job_queue, output_queue, logger_queue, word_set, synonyms):
	try:
		all_words = word_set + list(synonyms.keys())
		all_words = set([word.lower() for word in all_words])

		while True:
			text = job_queue.get()
			if text is None:
				break
			text = text.lower()
			found_words = []

			#text_words = nltk.word_tokenize(text) #text.split()
			#logger_queue.put(text_words)
			#ngrams = set(text_words)
			#for n in range(1, n_max+1):
			#	ngrams.update({space_delimiter.join(text_words[j:j+n+1]) for j in range(len(text_words)-n)})
			#found_words = ngrams.intersection(all_words)
			#tokens_of_text = nltk.word_tokenize(text)
			
			for word in all_words:
				# ice cream -> ['ice', 'cream']
				#tokens_of_word = nltk.word_tokenize(word) 
				
				# really simple ' word ' in text -> problem with punctuation... -> should tokenize -> problem more than one word words 
				if f' {word} ' in text:
					main_words = []

					# synonym tic-tac-toe belongs to word tick-tack-toe (which is not present in wikidumps)
					if word in synonyms:
						main_words += synonyms[word]

					# bike is a main word itself but also synonym of motobike -> main words: ['bike', 'motorbike']
					if word in word_set:
						main_words += [word]


					found_words.append((word, main_words))
					#main_word = word 
					#if word in synonyms:
					#	main_word = synonyms[word][0]
					#found_words[main_word].append(word)

			# Snake as synonym for snake 
			# main_word.lower() not in synonym.lower().split()
			# snakes as synonym for snake 
			# main_word.lower() != plural.lower() and main_word.lower() != synonym.lower():
								
			if found_words:
				output_queue.put((text, found_words))
	except Exception as e:
		logger_queue.put(f'Exc: {e}')
				
def write_results_to_output(output_queue, logger_queue, out_path, job_queue):
	counter = 0
	with open(out_path, 'w') as output_file:
		while True:
			record = output_queue.get()
			if record is None:
				break
			text, found_words = record
			text = text.replace('\n', '')
			output_file.write(';'.join([str(counter), json.dumps(found_words), text]))
			output_file.write('\n')
			counter += 1
			#if counter % 10000 == 0:
			#	logger_queue.put(f'DONE {counter} of {job_queue.qsize()}')

def iterate_corpus(path_to_corpus, wikidump=True, logger=None):
	if wikidump:
		folders = os.listdir(path_to_corpus)
		for i, folder in enumerate(folders):
			logger.debug(f'Read folder {folder} {i}/{len(folders)}')
			files = os.listdir(pjoin(path_to_corpus, folder))
			for file in files:
				for line in utils.file_to_lines(pjoin(path_to_corpus, folder, file)):
					mapping = json.loads(line)
					text = mapping['text']

					if text:
						# wikidump converts whole article into one text with breaks between sections
						sections = text.split('\n')
						for section in sections:
							# dont use section headlines
							if len(section.split()) > 3:
								yield section
	else:
		for line in utils.file_to_lines(path_to_corpus):
			yield line 

def find_texts(path_to_corpus, word_set, synset_mapping, logger, occurences_path, n_jobs, wikidumps=True):
	# Iterate thorugh corpus, find text with word
	m = Manager()
	output_queue = m.Queue()
	job_queue = m.Queue()
	logger_queue = m.Queue()

	lp = threading.Thread(target=utils.logger_thread, args=(logger_queue,))
	lp.start()

	op = Process(target=write_results_to_output, args=(output_queue, logger_queue, occurences_path, job_queue))
	op.start()

	jobs = []
	for i in range(n_jobs):
		logger.debug(f'FIND: start job {i}')
		job = Process(target=find_text, args=(job_queue, output_queue, logger_queue, word_set, synset_mapping))
		job.daemon = True  # only live while parent process lives
		job.start()
		jobs.append(job)

	logger.debug('FIND: start reading corpus')
	for text in iterate_corpus(path_to_corpus, wikidumps, logger):
		job = text
		job_queue.put(job)

	#find_text(job_queue, output_queue, logger_queue, word_set, synset_mapping)
	n = job_queue.qsize()
	logger.debug(f'Number of jobs: {n}')
	for _ in jobs:
		job_queue.put(None)

	output_queue.put(None)
	logger_queue.put(None)

	logger.debug('FIND: wait for jobs to finish')
	for w in jobs:
		w.join()

	# wait for it to finish
	logger.debug('FIND: wait for output job to finish')
	op.join()

	logger.debug('FIND: wait for logger job to finish')
	lp.join()

	logger.debug('FIND: done')

def word_to_corp_ind(sentences, word_set, synonyms={}, space_delimiter=' ', n_jobs=1, logger=None, out_path=''):
	unique_words = list(set(word_set))
	# TODO memory 50 GB too much array_split
	if n_jobs == 1:
		splitted_mappings = [sentences]
	else:
		splitted_mappings = np.array_split(sentences, n_jobs)
	
	number_of_sentences_per_batch = [len(batch) for batch in splitted_mappings]

	p = Pool(n_jobs)
	logger.debug('start jobs')
	def callback(e):
		logger.debug(e)
	
	jobs = [p.apply_async(run, (job_sentences, unique_words, synonyms, i, number_of_sentences_per_batch, out_path), error_callback=callback) for i, job_sentences in enumerate(splitted_mappings)]
	p.close()
	p.join()
	logger.debug('jobs done')

	found_words = defaultdict(lambda: 0)
	
	for i in range(n_jobs):
		job_out_path = pjoin(out_path, f'{i}.txt')
		with open(job_out_path, 'r') as anno_file:
			lines = anno_file.readlines()
			for line in lines:
				sen_ind, mapping = line.split(';')
				for word in json.loads(mapping).keys():
					if word in word_set:
						found_words[word] += 1
	

	found_words_set = set(found_words.keys())
	word_set = set(word_set)
	missing_words = word_set.difference(found_words_set)
	logger.debug(f'Found {len(found_words_set)} words - {len(missing_words)} missing')
	logger.debug(missing_words)
