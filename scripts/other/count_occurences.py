from collections import defaultdict
import pandas as pd 
from DecontextEmbeddings.scripts.transformer_embeddings.utils import file_to_lines
from collections import defaultdict
import json 

things_words = set(pd.read_csv('DecontextEmbeddings/data/things/item_names.tsv', sep='\t')['Word'])
counter = defaultdict(lambda: 0)

print(f'number of things words: {len(things_words)}')
print('start')
for line in file_to_lines('data/thinga/wikidumps/limited_occurences.txt'):
    words = json.loads(line.split(';')[0])
    for word in words:
        for main_word in word[1]:
            if main_word in things_words:
                counter[main_word] += 1

print(counter['showercap'])
print(counter['shower cap'])

df = pd.DataFrame({'word': list(counter.keys()), 'counts': list(counter.values())})
print(df['counts'].describe())
print(df.sort_values('counts', ascending=False).head(10))
print(df.sort_values('counts', ascending=True).head(30))

print(set(things_words).difference(set(df['word'])))