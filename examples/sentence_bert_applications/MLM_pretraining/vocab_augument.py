#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:45:43 2022

@author: chuang
"""

## vacabulary augumentation 

import os, sys 
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
#import config
from transformers_domain_adaptation import VocabAugmentor
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from arguments import vocab_aug_args
from utils import list2txt
#from pathlib import Path



#%%

if __name__ == "__main__":
    ## global path
    args = vocab_aug_args()
    corpus_small = os.path.join(args.input_files_folder,'raw.txt')
    VOCAB_OUT = os.path.join(args.model_folder,'imf_vocab_aug_500.txt')
    #AUG_MODEL_OUT = os.path.join(args.model_folder,args.model_checkpoint+"_aug")
    
    ## load target tokenizer 
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint,cache_dir= args.cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)
    except:
        ## if can not access internet, load from local 
        base_local_checkpoint = os.path.join(args.model_folder,args.model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(base_local_checkpoint,cache_dir= args.cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(base_local_checkpoint)

    ## set a new vocab size limit         
    org_vocab_size = tokenizer.vocab_size
    target_vocab_size = org_vocab_size + 500  # add only 500 vocabsize
 
    augmentor = VocabAugmentor(
        tokenizer=tokenizer,   ## the domain_adaptation package require tokenizer to be a older version. may need another environment
        cased=False,    # change this based on your tokenizer
        target_vocab_size=target_vocab_size
    )

    # Obtain new domain-specific terminology based on the fine-tuning corpus
    new_tokens = augmentor.get_new_tokens(corpus_small)
    new_tokens = [nt for nt in new_tokens if len(nt)>2] 
    
    list2txt(new_tokens,VOCAB_OUT)
    
    
    
    # ## adjust tokenizer and model dimention 
    # print('original vocab size: {}'.format(org_vocab_size))
    # tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    # print('new vocab size: {}'.format(len(tokenizer)))
    
    # ## save adjusted model and tokenizers 
    # tokenizer.save_pretrained(AUG_MODEL_OUT)
    # model.save_pretrained(AUG_MODEL_OUT)