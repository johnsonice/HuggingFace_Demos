#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:51:44 2022

@author: huang
"""
# Import required packages
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer,DataCollatorWithPadding,TrainingArguments
from datasets import load_dataset,load_metric
import config
import os

## modeify the head of someone's fintuned model : use the ignore mismatched size argument
## https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/19 

#%%

# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels,average=None)["f1"] 
   if isinstance(f1, np.ndarray):
       f1 = f1.tolist()  ## add this as nd array is not serializable when exporting 
   # print(load_metric("f1").inputs_description) ## print out descriptions 
   return {"accuracy": accuracy, "f1": f1}

#%%

if __name__ == "__main__":
    
    N_CPU = 6 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL_OUTDIR = os.path.join(config.data_folder,'news_classification','models')
    
    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## load pretrained and fintuned model with a ramdom initialized classifer layer
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=4,ignore_mismatched_sizes=True)
    trainer = Trainer(model=model)
    print(model.config) ## original classifier 
    
    #%%
    ## load an sample dataset 
    raw_datasets = load_dataset("ag_news")
    print(raw_datasets.keys())
    print(raw_datasets['train'][0])
    
    small_dataset_train = raw_datasets['train'].select(range(5000))
    small_dataset_test = raw_datasets['test'].select(range(1000))
    
    
    #%%
    ## tokenize data and prepare for dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True) #padding="max_length", ; normally you want to pad to global max length
                                                            # here we will use data collector later which will automately pad based on 
                                                            # longest of the batch 
    
    small_dataset_train = small_dataset_train.map(tokenize_function, batched=True,num_proc=N_CPU) 
    small_dataset_test = small_dataset_test.map(tokenize_function, batched=True,num_proc=N_CPU)
    small_dataset_train= small_dataset_train.remove_columns(['text'])
    small_dataset_test= small_dataset_test.remove_columns(['text'])
    
    print(small_dataset_train[0]) ## print one for visual inspection
    
    
    #%%
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(output_dir=MODEL_OUTDIR,
                                   #evaluation_strategy="epoch",
                                   evaluation_strategy="steps",
                                   eval_steps=200,
                                   logging_steps =200,          ## show eval results
                                   learning_rate=2e-5,
                                   per_device_train_batch_size=8,
                                   per_device_eval_batch_size=8,
                                   num_train_epochs=5,
                                   weight_decay=0.4, ## wd regularizor, usually a very small number as additional weight penality
                                   save_steps=200,
                                   load_best_model_at_end=True, ## only save and load best model
                                   save_total_limit = 1,        ## only save one checkpoint
                                   seed=RANDOM_SEED)  
    ## set up trainer 
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=small_dataset_train,
       eval_dataset=small_dataset_test,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )
        
    
    ## train 
    t = trainer.train()
    x = trainer.evaluate() 
    print(t,x)
    trainer.save_model(MODEL_OUTDIR)
        
    