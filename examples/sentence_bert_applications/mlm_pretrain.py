#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:03:50 2022

@author: chengyu

## domain adaption for MLM models 

"""

import os, sys 
sys.path.insert(0,'../../libs')
#from hf_utils import train_val_test_split

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
from datasets import load_dataset,load_from_disk
from transformers import DataCollatorForLanguageModeling

import collections
import numpy as np
from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer

import config

#%%
def try_mlm(text,tokenizer,model):
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

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

#%%
if __name__ == "__main__":
    ## specify path 
    MODEL_OUTDIR = os.path.join(config.model_folder,'sentence_bert_mlm')
    data_path= os.path.join(config.data_folder,'Data/sentence_bert/pre_training_raw_data','IMF_Documents_2018.txt')
    DA_OUTDIR= os.path.join(config.data_folder,'Data/sentence_bert/pre_training_processed_ds')
    model_checkpoint = "distilbert-base-uncased"
    #%%
    DATA_PROCESS_FLAG = False
    
    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    if DATA_PROCESS_FLAG:
        ##process data into datastes 
        raw_dataset = load_dataset('text', data_files=data_path) ## default split is 'train'
        raw_ds = raw_dataset['train'].train_test_split(test_size=0.01,
                                                  shuffle=True,
                                                  seed=42)
        
        ##tokenize data // here, it is ok to have longer sequences as we will chunk them in the next step
        tokenized_datasets = raw_ds.map(
            tokenize_function, batched=True, remove_columns=["text",]
        )
        ## chunk data into fixed length text sequences
        lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        lm_datasets.save_to_disk(DA_OUTDIR)
        print(lm_datasets)
    else:
        ## load an sample dataset 
        lm_datasets = load_from_disk(DA_OUTDIR)
    
    #%%
    ## we use whole word maksing here 
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    data_collator = whole_word_masking_data_collator
    remove_unused_columns=False ## if use whole word masking, set this to be false 
    #%%
    
    ## Load Model 
    print('######### Initiate Model ######')
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    distilbert_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")
    
    batch_size = 64
    # Show the training loss with every epoch or fix steps 
    #logging_steps = len(lm_datasets["train"]) // batch_size
    logging_steps = 256
    
    model_name = model_checkpoint.split("/")[-1]+'imf_mlm'
    #model_out_dir = os.path.join(config.data_folder,'MLM','models',"{}-finetuned-imdb".format(model_name))
    
    training_args = TrainingArguments(
        output_dir=MODEL_OUTDIR,
        overwrite_output_dir=True,
        num_train_epochs = 3,
        evaluation_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=10*logging_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #push_to_hub=False,  ## just run locally for now 
        fp16=True,
        remove_unused_columns=remove_unused_columns
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )
    
    #%%
    ## see the original perplexity before training
    print(trainer.evaluate())
    
    #%%
    trainer.train()
    
    
    
    
    