import torch 
import logging 
import json 
import os 
from os.path import join as pjoin 

def extract_embeddings(job_queue, output_queue, model_name, device, logger_queue, i) -> List[torch.Tensor]:
    """Aggregate hidden representations across L layers and subsequently pool subword reps if necessary."""
    logger = logging.getLogger(f'{i}')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{i}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    model = utils.load_model(model_name, device)
    tokenizer = utils.load_tokenizer(model_name)

    torch.set_num_threads(1)
        
    while True:
        record = job_queue.get()
        if record is None:
            print(record)
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            logger_queue.put('job: ' + current_time)
            break
        try:
            text, mapping = record
            processed = utils.tokenize_sentence(tokenizer, text, True, True, False).to(model.device)
            number_of_tokens = len(processed['input_ids'][0])
            if number_of_tokens > tokenizer.model_max_length:
                continue 
            all_token_embeddings = utils.get_hiddens(model, processed)

            mappings = json.loads(mapping)
                      
            for mapping in mappings:
                if len(mapping) == 3:
                    word_found_in_sentence, main_words, synsets = mapping
                else:
                    word_found_in_sentence, main_words = mapping 
                    synsets = None 

                logger.debug(f'Word found in sentence: {word_found_in_sentence} Main Words: {main_words} Synsets: {synsets} ')
                word_embeddings, _ = utils.get_embeddings_of_word(word_found_in_sentence, processed, all_token_embeddings, tokenizer, logger)
                # can happen that more word embeddings than synsets are found as GPT has higher input length
                # than BERT, and BERT is used to find synsets
                if synsets:
                    word_embeddings = word_embeddings[:len(synsets)]

                for i, word_embedding in enumerate(word_embeddings):
                    synset = ['', '']
                    if synsets:
                        synset = synsets[i]
                    result = (word_found_in_sentence, main_words, synset, word_embedding.cpu().numpy())
                    output_queue.put(result)

        except Exception as e:
            e = traceback.format_exception(value=e, tb=e.__traceback__,etype=Exception)
            logger_queue.put(f'Exc: {e} in job {i}')
        
def write_results_to_output(output_queue, out_path, n_layers):
    file_handlers = []
    for i in range(n_layers):
        layer_out_path = pjoin(out_path, str(i))
        if not os.path.exists(layer_out_path):
            os.makedirs(layer_out_path)
        handler = open(pjoin(layer_out_path, 'extractions.txt'), 'w')
        file_handlers.append(handler)

    while True:
        record = output_queue.get()
        if record is None:
            print(record)
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)
            break
        word_found, main_words, synset_pair, embedding = record		
        synset, concept_id = synset_pair

        for layer in range(embedding.shape[0]):	
            output_file = file_handlers[layer]
            embedding_str = ' '.join(str(value) for value in embedding[layer])
            main_words_str = ','.join(main_words)
            output_file.write(';'.join([word_found, main_words_str, synset, concept_id, embedding_str]))
            output_file.write('\n')
        
        
def queue_watcher(job_queue, logger_queue, output_queue):
    while True:
        import time 
        time.sleep(300)
        for queue, queue_name in [(job_queue, 'Job'), (logger_queue, 'Logger'), (output_queue, 'Output')]:
            number_jobs = queue.qsize()
            logger_queue.put(f'QUEUE: {queue_name}: {number_jobs} jobs left')
