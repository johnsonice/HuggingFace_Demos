#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:51:44 2022

@author: huang
"""
# Import required packages
import os,sys
sys.path.insert(0,'../../libs')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer,DataCollatorWithPadding,TrainingArguments
from datasets import load_dataset,load_from_disk
import config
from hf_utils import compute_acc_f1
## modeify the head of someone's fintuned model : use the ignore mismatched size argument
## https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/19 

## hyperparameter-search in trainer
# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785?page=3
# https://huggingface.co/blog/ray-tune
#%%

if __name__ == "__main__":
    
    
    param_dict = {"attention_probs_dropout_prob": 0.2, 
                         "hidden_dropout_prob": 0.3, 
                         "learning_rate": 2e-05, 
                         "per_device_train_batch_size": 2, 
                         "gradient_accumulation_steps": 2, 
                         "weight_decay": 0.05, 
                         "label_smoothing_factor": 0.1, 
                         "metric_for_best_model": "accuracy"}

    
    N_CPU = config.N_CPU 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification')
    DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    
    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    #model_name = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## load pretrained and fintuned model with a ramdom initialized classifer layer
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=3,
                                                               ignore_mismatched_sizes=True,
                                                               attention_probs_dropout_prob=0.2, ##change drop out 
                                                               hidden_dropout_prob=0.3) ##change drop out 
    #trainer = Trainer(model=model)
    print(model.config) ## original classifier 
    
    ## load an sample dataset 
    input_dataset = load_from_disk(DATASET_DIR)
    print(input_dataset)
    input_dataset = input_dataset.remove_columns('text') ## remove raw text for training
    
    ## set up data collactor for dynamic padding in batch 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    ## set up training args 
    training_args = TrainingArguments(output_dir=MODEL_OUTDIR,
                                   #evaluation_strategy="epoch",
                                   evaluation_strategy="steps",
                                   eval_steps=30,
                                   logging_steps =30,          ## show eval results
                                   learning_rate=2e-05,
                                   per_device_train_batch_size=2,
                                   per_device_eval_batch_size=2,
                                   gradient_accumulation_steps=2,
                                   num_train_epochs=10,
                                   warmup_steps=25,
                                   weight_decay=0.05,    ## wd regularizor, usually a very small number as additional weight penality
                                   label_smoothing_factor=0.1,
                                   save_steps=30,
                                   load_best_model_at_end=True, ## only save and load best model
                                   metric_for_best_model='accuracy',
                                   save_total_limit = 4,        ## only save one checkpoint
                                   seed=RANDOM_SEED)  
    ## set up trainer 
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=input_dataset['train'],
       eval_dataset=input_dataset['val'],
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_acc_f1,
    )
        
    #%%
    ## train 
    t = trainer.train()
    x = trainer.evaluate() 
    print('Training; Evaluation Results:')
    print(t,x)
    print('Test Results:')
    test_res = trainer.evaluate(input_dataset['test'])
    print(test_res)
    #%5
    trainer.save_model(MODEL_OUTDIR)
        
    