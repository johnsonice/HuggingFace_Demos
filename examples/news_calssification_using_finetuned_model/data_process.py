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

def load_split_climate_data_2label(raw_data_path):
    df = pd.read_excel(raw_data_path)[['title','body','Rating']]
    
    df['Rating'] = df['Rating'] .replace(0,1)   # change nutual to Positive 
    df['Rating'] = df['Rating'] .replace(-1,0)  # change negative to 0 
    
    print('label description: \n{}'.format(df['Rating'].value_counts(normalize=True)))
    ds = Dataset.from_pandas(df)
    ds= ds.rename_column("Rating","label")
    new_features = ds.features.copy()
    new_features['label'] = ClassLabel(names=['Decrease Risk','Increase Risk'])
    ds = ds.cast(new_features)
    ## increate val_test to 40%
    ds = train_val_test_split(ds,val_test_ratios=0.3,label_col='label',RANDOM_SEED=42)
    return ds

def merge_columns(example):
    example["text"] = '{} ; {}'.format(example['title'],example['body'])
    return example


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


#%%
if __name__ == "__main__":
    
    TOKENIZE_FLAG = True
    file_name = 'Climate_training_paragraphs_v2.xlsx'#_overweight.xlsx'
    raw_data = os.path.join(config.data_folder,'Data','climate_news',file_name)
    
    #baseline_ds_dir = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    baseline_ds_dir = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset','baseline_2label')
    
    if TOKENIZE_FLAG:
        model_name = "siebert/sentiment-roberta-large-english"
        #model_name = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ## load climate data into datasets and split it by train val test
    #ds = load_split_climate_data(raw_data)
    ds = load_split_climate_data_2label(raw_data)
    
    ## Baseline scenario 
    ## merge title with body as input text and do classification 

    basline_ds = ds.map(merge_columns,remove_columns=['title','body'],batched=False,num_proc=6)
    ## tokenize data 
    if TOKENIZE_FLAG:
        basline_ds = basline_ds.map(tokenize_function,batched=True,num_proc=6)
    ## export to disk
    basline_ds.save_to_disk(baseline_ds_dir)