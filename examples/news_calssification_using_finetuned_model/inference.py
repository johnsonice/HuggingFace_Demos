#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:15:57 2022

@author: chengyu
"""

## inference 
## https://discuss.huggingface.co/t/using-trainer-at-inference-time/9378/7

import os,sys
sys.path.insert(0,'../../libs')
from transformers import Trainer,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments
from datasets import load_dataset,load_from_disk
import config
from hf_utils import compute_acc_f1

#%%
if __name__ == "__main__":    

    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification')
    DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTDIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTDIR)
    model.eval()
    
    
    #%%
    text = ['european public goods must be paid for by the european taxpayer. the key reform as far as revenue is concerned is to grow the proportion of the autonomous â€˜own resourcesâ€™ of the eu (now only 25 per cent) and thereby to reduce direct national contributions based on gni. the eu institutions should engage with each other in developing a coherent package of new streams of revenue, including eu taxes on carbon emissions, financial transactions and aviation, as well as a distinctive tranche of vat. the new scheme must be ready to be introduced in time to fund the new medium-term multi-annual financial framework due in 2013. ']
    encoding = tokenizer(text,return_tensors='pt')
    
    # forward pass
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)
    print(outputs)
    #%%
    input_dataset = load_from_disk(DATASET_DIR)
    # arguments for Trainer
    test_args = TrainingArguments(
        output_dir = MODEL_OUTDIR,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = 4,   
        dataloader_drop_last = False    
    )
    
    # init trainer
    trainer = Trainer(
                  model = model, 
                  args = test_args, 
                  tokenizer=tokenizer,
                  compute_metrics = compute_acc_f1)
    #%%
    #test_dataset=input_dataset['test'].remove_columns('text')
    #%%
    test_results = trainer.predict(input_dataset['train'])