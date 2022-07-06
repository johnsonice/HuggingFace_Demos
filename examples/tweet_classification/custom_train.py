#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:30:13 2022

@author: chengyu
"""

import sys,os
sys.path.insert(0, '../../libs')
from utils import load_jsonl
import config
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train import compute_metrics

## import transformer packages 
import datasets
from datasets import Dataset,load_metric,logging as dataset_logging
from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,logging as transformers_logging
from transformers import TrainingArguments, Trainer
from transformers import get_scheduler
transformers_logging.set_verbosity_error() ## make it less verbose
dataset_logging.set_verbosity_error() ## make it less verbose
print('gpu available : {}'.format(torch.cuda.is_available()))

#%%
if __name__ == "__main__":
    
    ## set up global params 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL = "bert-base-cased" ## this need to be the same as your tokenizer 
    N_CPU = 6 
    out_dir = os.path.join(config.data_folder,'tweet_classfier')
    tokenized_outdir = os.path.join(config.data_folder,'tweets_dataset','tokenized','tokenized_data')
    
    #load preprocessed data with bert tokenizer
    tokenized_datasets = datasets.load_from_disk(tokenized_outdir)
    ## split out train and test 
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    
    #%%
    ### customized trianing loop 
    train_dataloader = DataLoader(
                        tokenized_datasets["train"], shuffle=True, 
                        batch_size=8, collate_fn=data_collator
                        )   
    eval_dataloader = DataLoader(
                        tokenized_datasets["test"], batch_size=8, 
                        collate_fn=data_collator
                        )
    
    ## quick test - see shape of one batch 
    for batch in train_dataloader:
        break
    print({k: v.shape for k, v in batch.items()})
    
    #%%
    ## load model 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    ## test see if the structue is correct : do one forward pass
    print('Do one forward pass for checking')
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    #%%
    ## set up training params 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)
    
    #%%
    ## training loop 
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(device)
    
    #%%
    ### for distributed training please follow https://huggingface.co/course/chapter3/4?fw=pt
    ## training 
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    ## evaluation 
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    print(metric.compute())
    
    