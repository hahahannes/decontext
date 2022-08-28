import pandas as pd 
from nltk.corpus import wordnet
import numpy as np 

# Idea: for each THINS concept/word get the synsset from the table 
# then get all synonyms not only the ones from the list (to be sure to get all)
# then for each potential synonym get synsets
# only consider as true synyms when top1 most frequent synset of word and synonym are same

df = pd.read_csv('things_concepts.tsv', sep='\t')

def word_are_syn(word1_synset, word2, pos):
    top_synsets = [syn for syn in wordnet.synsets(word2, pos=pos)][:1]
    top_synsets = [syn.name() for syn in top_synsets]
    if type(word1_synset) != str and np.isnan(word1_synset):
        return False
    return word1_synset in top_synsets

def get_all_possible_syns(word, pos):
	return list(set([lemma for ss in wordnet.synsets(word, pos=pos) for lemma in ss.lemma_names()]))

def get_two_word_variants(word):
	normalized_word = word.replace('-', ' ').replace('_', ' ')
	normalized_words = normalized_word.split(' ')
	if len(normalized_words) > 1:
		variants = ['-'.join(normalized_words), ' '.join(normalized_words), ''.join(normalized_words)]
		variants = [variant for variant in variants if variant != word]
	else:
		variants = [word]
	
	return variants
	
def get_synonyms(word, pos=None, synset=None):
	word = word.lower()
	words = get_two_word_variants(word)
	syns = []
	

	for variant_word in words:
		variant_syns = []
		if pos:
			variant_syns += get_all_possible_syns(variant_word, pos)
		elif synset:
			all_possible_syns = get_all_possible_syns(variant_word, pos)
			variant_syns += [synonym for synonym in all_possible_syns if word_are_syn(synset, synonym, pos)]

		# TODO synonym `fastfood` for `fast food`
		# TODO two word words -> combined/space sep/minus separated `t-shirt` and `tshirt` `t shirt`
		
		# only replace _ from wordnet words but not - as there are words like t-shirt
		variant_syns.append(variant_word)
		variant_syns = [syn.replace('_', ' ').lower() for syn in variant_syns]
		# all lowercase -> remove same synonyms like Snake (snake) vs snake
		# remove synoyms that contain the word with spaces -> lead to duplicate embeddings like army tank vs. tank
		variant_syns = [syn for syn in variant_syns if word not in syn.split(' ')]
		
		variant_syns = [syn for syn in variant_syns if word != syn]
		
		if 'iceskate' == word:
			variant_syns.append('ice skate')
		elif 'videogame' == word:
			variant_syns.append('video game')
		elif 'bathmat' == word:
			variant_syns.append('bath mat')
			
		syns += variant_syns
	syns = list(set(syns))
	return syns

new = []

for row in df.iterrows():
	row = row[1]
	word = row['Word']
	synset = row['Wordnet ID4']
	true_syns = get_synonyms(word, pos=None, synset=synset)
    
	#all_synonyms = list(set([lemma for ss in wordnet.synsets(word) for lemma in ss.lemma_names()]))
    #print(all_synonyms)
    #true_syns = []
    #for possible_syn in all_synonyms:
    #    if possible_syn != word:
    #        is_syn = word_are_syn(synset, possible_syn, wordnet.NOUN)
    #        if is_syn:
    #            true_syns.append(possible_syn)
    
	new.append({'id': row.uniqueID, 'Word': row['Word'], 'synset': row['Wordnet ID4'], 'Wordnet Synset': ','.join(true_syns)})

pd.DataFrame(new).to_csv('things_syns.csv', index=False)




#p = inflect.engine()

def get_singular_or_plural_version(word, pos):
	# problems: detect if word is alredy plural / check singular depends on POS
	if pos == wn.NOUN:
		if p.singular_noun(word):
			return p.plural(word)
		else:
			return p.singular(word)
	elif pos == wn.ADJ:
		pass