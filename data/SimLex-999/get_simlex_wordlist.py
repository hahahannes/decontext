import pandas as pd

simlex_file = './SimLex-999.txt'

df = pd.read_csv(simlex_file, delimiter='\t')
print(df)
words_1 = df['word1'].tolist()
words_2 = df['word2'].tolist()

final_word_list = list(set(words_1 + words_2))

out_df = pd.DataFrame(columns=['Word', 'Wordnet Synset'])
out_df['Word'] = final_word_list
out_df['Wordnet Synset'] = final_word_list

out_df.to_csv('Simlex-998-wordlist.txt', sep='\t', index=False)

