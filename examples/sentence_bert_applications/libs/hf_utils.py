#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:40:47 2022

@author: chengyu
"""

## hugginface module utiles 
#import torch
#import pandas as pd
import numpy as np
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer,DataCollatorWithPadding,TrainingArguments
from datasets import load_metric,DatasetDict
import gc, torch 

def clear_memory(verbose=False):
    #stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.') 

def compute_acc_f1(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels,average=None)["f1"] 
   if isinstance(f1, np.ndarray):
       f1 = f1.tolist()  ## add this as nd array is not serializable when exporting 
   # print(load_metric("f1").inputs_description) ## print out descriptions 
   global_f1 = sum(f1)/len(f1)
   return {"accuracy": accuracy, "f1": f1, "global_f1":global_f1}

def compute_acc(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_val_test_split(ds,val_test_ratios=0.3,label_col='label',RANDOM_SEED=42):
    '''
    Parameters
    ----------
    ds : Dataset object
        DESCRIPTION.
    val_test_ratios : float, optional
        DESCRIPTION. The default is 0.3.
    label_col : string, optional
        DESCRIPTION. label column for stratified sampling; The default is 'lable'.
    RANDOM_SEED : int, optional
        DESCRIPTION. The default is 42.
        
    Returns
    -------
    res_ds : Dataset object ; Dataset dict
    '''
    val_test_split = 0.5
    splited_ds = ds.train_test_split(test_size=val_test_ratios,shuffle=True,
                                     seed=RANDOM_SEED,
                                     stratify_by_column=label_col)
    eval_test_ds = splited_ds['test'].train_test_split(test_size=val_test_split,shuffle=True,
                                                       seed=RANDOM_SEED,stratify_by_column=label_col)
    
    res_ds = DatasetDict({'train':splited_ds['train'], 'val':eval_test_ds['train'],'test':eval_test_ds['test']})
    return res_ds

# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}