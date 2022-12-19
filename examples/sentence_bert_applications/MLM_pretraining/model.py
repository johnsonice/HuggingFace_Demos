#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:49:39 2022

@author: chuang
"""

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

def txt2list(f_p):
    with open(f_p,'r',encoding='utf8') as fp:
        contents = fp.readlines()
        contents = [c.strip('\n') for c in contents]
    return contents

def get_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_config(config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    if args.additional_vocab_path:
        print('expand original vocabulary for tokenization; load from {}'.format(args.additional_vocab_path))
        new_tokens = txt2list(args.additional_vocab_path)
        print('original vocab size: {}'.format(tokenizer.vocab_size))
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print('new vocab size: {}'.format(len(tokenizer)))
    
    return model,tokenizer