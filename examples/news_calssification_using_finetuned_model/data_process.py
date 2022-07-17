#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:42:42 2022

@author: chengyu
"""

### data process
import os, sys 
sys.path.insert(0,'../../libs')
import pandas as pd 
import config
from datasets import Dataset,load_dataset,ClassLabel,DatasetDict
from transformers import AutoTokenizer
from hf_utils import train_val_test_split
#%%

def load_split_climate_data(raw_data_path):
    df = pd.read_excel(raw_data_path)[['title','body','Rating']]
    df['Rating'] = df['Rating'] .replace(-1,2)
    print('label description: \n{}'.format(df['Rating'].value_counts(normalize=True)))
    ds = Dataset.from_pandas(df)
    ds= ds.rename_column("Rating","label")
    new_features = ds.features.copy()
    new_features['label'] = ClassLabel(names=['Neutral','Increase Risk','Decrease Risk'])
    ds = ds.cast(new_features)
    ds = train_val_test_split(ds,val_test_ratios=0.3,label_col='label',RANDOM_SEED=42)
    return ds
#%%
if __name__ == "__main__":
    
    TOKENIZE_FLAG = True
    
    raw_data = os.path.join(config.data_folder,'Data','climate_news','Climate_training_paragraphs.xlsx')
    baseline_ds_dir = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    
    if TOKENIZE_FLAG:
        model_name = "siebert/sentiment-roberta-large-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ## load climate data into datasets and split it by train val test
    ds = load_split_climate_data(raw_data)
    
    ## Baseline scenario 
    ## merge title with body as input text and do classification 
    def merge_columns(example):
        example["text"] = '{} ; {}'.format(example['title'],example['body'])
        return example
    basline_ds = ds.map(merge_columns,remove_columns=['title','body'],batched=False,num_proc=6)
    ## tokenize data 
    if TOKENIZE_FLAG:
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True)
        basline_ds = basline_ds.map(tokenize_function,batched=True,num_proc=6)
    ## export to disk
    basline_ds.save_to_disk(baseline_ds_dir)