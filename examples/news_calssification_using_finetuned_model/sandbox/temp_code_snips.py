#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:50:37 2022

@author: chengyu
"""

## temp code snips

## load AG news data set 
    raw_datasets = load_dataset("ag_news")
    print(raw_datasets.keys())
    print(raw_datasets['train'][0])
    small_dataset_train = raw_datasets['train'].select(range(5000))
    small_dataset_test = raw_datasets['test'].select(range(1000))
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