#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
## txt data to tokens for MLM models 
"""
import os, sys 
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
#from hf_utils import train_val_test_split
#import wandb
#from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
#import torch
from datasets import load_dataset,load_from_disk
#from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
from transformers import default_data_collator
#import config
from utils import get_all_files,txt2list
from arguments import tokenize_args
from multiprocessing import cpu_count

#%%
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        ## get world id for whole word masking 
    return result

def group_texts(examples,chunk_size=128):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column, for MLM, we put labels as inputs ids 
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features,wwm_probability=0.2):
    for feature in features: 
        word_ids = feature.pop("word_ids")  ## data collator does not take word_ids
 
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)  ## create a whold word to token ids map

        mask = np.random.binomial(1, wwm_probability, (len(mapping),)) ## create a 0 , 1 list with x prob to be 1
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels) 
        for word_id in np.where(mask)[0]: ## loop through all masked idx 
            word_id = word_id.item()
            for idx in mapping[word_id]: ## use mapping to get token indexes 
                new_labels[idx] = labels[idx]  ## in lables, masked token will be actual token index
                input_ids[idx] = tokenizer.mask_token_id   ## masked tokens will be replaced with mask token id
    
    return default_data_collator(features)

if __name__ == "__main__":
    
    N_cpu = 8 #cpu_count() - 8
    args = tokenize_args()
    print(args)
    
    MODEL = args.model_checkpoint
    DA_CACHE = args.cache_dir
    DA_OUTDIR= args.ds_out_folder #os.path.join(config.data_folder,'Data/sentence_bert/mlm_pre_training_processed_{}'.format(MODEL))
    #data_path= os.path.join(config.data_folder,'Data/sentence_bert/pre_training_raw_data','IMF_Documents_2018.txt')
    IN_DA_FOLDER = args.input_files_folder
    data_files = get_all_files(IN_DA_FOLDER,'.txt')

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if args.vocab_aug:
        v_p = os.path.join(args.model_folder,'imf_vocab_aug_500.txt')
        print('expand original vocabulary for tokenization; load from {}'.format(v_p))
        new_tokens = txt2list(v_p)
        print('original vocab size: {}'.format(tokenizer.vocab_size))
        tokenizer.add_tokens(new_tokens)
        #model.resize_token_embeddings(len(tokenizer))
        print('new vocab size: {}'.format(len(tokenizer)))
        
    ##process data into datastes 
    try:
        raw_dataset = load_dataset('text', data_files=data_files,cache_dir=DA_CACHE) ## default split is 'train'
    except:
        local_script = os.path.join(args.data_folder,'local_scripts','text.py')
        raw_dataset = load_dataset(local_script, data_files=data_files,cache_dir=DA_CACHE)
    
    raw_ds = raw_dataset['train'].train_test_split(test_size=0.001,
                                                shuffle=True,
                                                seed=42)
    ##tokenize data // here, it is ok to have longer sequences as we will chunk them in the next step
    tokenized_datasets = raw_ds.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text",],
        num_proc = N_cpu
    )
    ## chunk data into fixed length text sequences
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    lm_datasets.save_to_disk(DA_OUTDIR)
    print(lm_datasets)

