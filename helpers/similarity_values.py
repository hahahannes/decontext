from helpers.things_evaluation.evaluate import load_behav, load_sorting
import pandas as pd 
import math

def get_ranks(df):
    def calc_corr(row):
        #return abs(row.human_rank - row.embedding_rank_w2v)
        return (row.human_rank - row.embedding_rank_w2v) ** 2

    def calc_corr_bert(row):
        #return abs(row.human_rank - row.embedding_rank_bert)
        return (row.human_rank - row.embedding_rank_bert) ** 2

    def calc_corr_gpt(row):
        #return abs(row.human_rank - row.embedding_rank_gpt)
        return (row.human_rank - row.embedding_rank_gpt) ** 2

    human = 'Human Judgement'
    w2v = 'Cosine Similarities (Word2Vec)'
    bert = 'Cosine Similarities (BERT)'
    gpt = 'Cosine Similarities (GPT-2)'

    if human in df.columns:
        df['human_rank'] = df[human].sort_values().rank()
    
    if w2v in df.columns:
        df['embedding_rank_w2v'] = df[w2v].sort_values().rank()
    
    if bert in df.columns:
        df['embedding_rank_bert'] = df[bert].sort_values().rank()
    
    if gpt in df.columns:
        df['embedding_rank_gpt'] = df[gpt].sort_values().rank()
    
    if 'embedding_rank_w2v' in df.columns:
        df['di^2_w2v'] = df.apply(calc_corr, axis=1)
    
    if 'embedding_rank_bert' in df.columns:
        df['di^2_decontext_bert'] = df.apply(calc_corr_bert, axis=1)

    if 'embedding_rank_gpt' in df.columns:
        df['di^2_decontext_gpt'] = df.apply(calc_corr_gpt, axis=1)

    return df

def calc_diffs(df):
    def calc_cosine_diff(row):
        return (row['embedding_cosine_w2v'] - row['embedding_cosine_decontext']) ** 2

    def calc_rank_diff(row):
        return (row['embedding_rank_w2v'] - row['embedding_rank_decontext']) ** 2

    def calc_diff_of_rank_diffs_to_human(row):
        return (row['di^2_w2v'] - row['di^2_decontext']) ** 2
        
    df['cosine_diff'] = df.apply(calc_cosine_diff, axis=1)
    df['squared_rank_diff_of_human_rank_diff'] = df.apply(calc_diff_of_rank_diffs_to_human, axis=1)
    df['squared_rank_diff'] = df.apply(calc_rank_diff, axis=1)
    return df

def get_spose_matrix():
    behav_sim = load_behav()
    sorting_df = load_sorting()
    things_df = pd.DataFrame(behav_sim, index=sorting_df['concept_id'], columns=sorting_df['concept_id'])
    return things_df

def rsa_matrix_to_pair_list(df):
    columns = list(df.columns)
    pair_rows = []

    for i, row in enumerate(df.iterrows()):
        word1 = row[0]
        for col in columns[i+1:]:
            pair_rows.append({'word1': word1, 'word2': col, 'sim': row[1][col]})

    return pd.DataFrame(pair_rows)