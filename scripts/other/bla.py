import DecontextEmbeddings.scripts.transformer_embeddings.utils as utils 
import torch 

model_name = 'sentence-transformers/all-distilroberta-v1'
tokenizer = utils.load_tokenizer(model_name)
model = utils.load_model(model_name, 'cpu')
#print(model)



def get_token_ranges(tokenizer, word, processed):        
    indices_of_matched_tokens = []

    if True:
        word_tokenized = utils.tokenize_sentence(tokenizer, word, False).to('cpu')
        word_tokens = tokenizer.convert_ids_to_tokens(word_tokenized['input_ids'][0], skip_special_tokens=True)
        print(word_tokens)
        # TODO version with space in the beginning as diffent encoding in gpt
        
        n_tokens = len(word_tokens)
        tokens_of_one_sentence = tokenizer.convert_ids_to_tokens(processed['input_ids'][0], skip_special_tokens=True)
        print(tokens_of_one_sentence)
        indices_of_matched_tokens = []
        for i in range(len(tokens_of_one_sentence)):
            start = None
            end = None
            match = True
            for m in range(n_tokens):
                if i+m <= len(tokens_of_one_sentence)-1 and tokens_of_one_sentence[i+m] == word_tokens[m]:
                    if m == 0:
                        start = i+m
                        if n_tokens == 1:
                            end = i+m
                    elif m == n_tokens-1:
                        end = i+m
                    continue
                else:
                    match = False  
                    start = None 
                    end = None
                    break 

            if match:
                start_end_indices = (start, end)
                indices_of_matched_tokens.append(start_end_indices)
                start = None 
                end = None
    return indices_of_matched_tokens

def get_string(sent, word):
    processed = utils.tokenize_sentence(tokenizer, sent, True, True, False)
    print(processed)
    offset_mappings = processed['offset_mapping'][0]
    offset_mappings = offset_mappings[torch.where(processed['special_tokens_mask'][0] == 0)]

    hiddens = utils.get_hiddens(model, processed)
    # hiddens are only for non special tokens
    print(hiddens.shape)

    token_occurences = get_token_ranges(tokenizer, word, processed)
    print(token_occurences)
    offset_mapping_of_sentence = offset_mappings
    for token_occurence in token_occurences:
        start_token_index = token_occurence[0]
        end_token_index = token_occurence[1]

        # offset mapping geht nicht mehr, da hier special tokens mit drin sind
        string_start = offset_mapping_of_sentence[start_token_index][0].item()
        string_end = offset_mapping_of_sentence[end_token_index][1].item()
        print(f'{sent[string_start:string_end]}')

get_string('hello this hello is', 'hello')
get_string('bla this this', 'this')


# BERT
# CLS ... SEP -> need indices without special tokens as hiddens from model are only for non special tokens
# ['[CLS]', 'hello', 'this', 'is', '[SEP]']

# GPT2
# ['Ghello', 'Gthis',...]

# GPT2 medium
# tokenizer macht leerzeichen zu G -> offset mapping - token matching inkludiert dann leerzeichen
# ['▁hello', '▁this', '▁is', '<sep>', '<cls>']

# problem gpt-2 has no special tokens and gpt2-medium has special tokens but different position than BERT
# -> skip special tokens to find sub tokens
# -> found indexes are based on non special tokens but model output hidden states has all tokens -> remove special tokens from there so that indexes match  