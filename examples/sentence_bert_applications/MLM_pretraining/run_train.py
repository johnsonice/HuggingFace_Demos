#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:55:24 2022

@author: chuang
"""
#%%
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling,default_data_collator
import collections
import numpy as np
import os 
from model import get_model
from data import get_data
from optimizer import get_optimizer
from arguments import ModelTrainingArguments
from transformers import HfArgumentParser
from trainer import train
from functools import partial
#import wandb

def whole_word_masking_data_collator(features,tokenizer,wwm_probability=0.2):
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
    

#%%
if __name__ == "__main__":
    print('...run train ...')
    parser = HfArgumentParser(
        ModelTrainingArguments,
    )
    args = parser.parse_args_into_dataclasses()[0]
    model, tokenizer = get_model(args)
    dataset = get_data(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    #data_collator = partial(whole_word_masking_data_collator,tokenizer=tokenizer)
    optimizer = get_optimizer(model,args)

    ## start training
    T = train(tokenizer, data_collator, dataset, model, optimizer, args)
    ## save last iteration of the model 
    T.save_model(args.output_dir)
    eval_res = T.evaluate() 
    print('Evaluation Results: {}'.format(eval_res))