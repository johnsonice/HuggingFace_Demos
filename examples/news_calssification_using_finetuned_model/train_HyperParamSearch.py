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
from datasets import load_dataset,load_from_disk,load_metric
import config
import numpy as np
from ray import tune
#from hf_utils import compute_acc_f1,compute_acc
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.tune.schedulers import ASHAScheduler

## modeify the head of someone's fintuned model : use the ignore mismatched size argument
## https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/19 

## hyperparameter-search in trainer
# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785?page=3
# https://huggingface.co/blog/ray-tune
#%%

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=3,
                                                               ignore_mismatched_sizes=True,
                                                               attention_probs_dropout_prob=0.3, ##change drop out 
                                                               hidden_dropout_prob=0.4, ##change drop out 
                                                               return_dict=True) 
    


def compute_acc(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


## define a evaluation function : here it is a simple accuracy 


if __name__ == "__main__":
    
    N_CPU = config.N_CPU 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification')
    DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    
    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## load pretrained and fintuned model with a ramdom initialized classifer layer
    # model = AutoModelForSequenceClassification.from_pretrained(model_name,
    #                                                            num_labels=3,
    #                                                            ignore_mismatched_sizes=True,
    #                                                            attention_probs_dropout_prob=0.3, ##change drop out 
    #                                                            hidden_dropout_prob=0.4) ##change drop out 
    # #trainer = Trainer(model=model)
    # print(model.config) ## original classifier 
    
    #%%
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
                                   learning_rate=1e-5,
                                   per_device_train_batch_size=2,
                                   per_device_eval_batch_size=2,
                                   #gradient_accumulation_steps=2,
                                   num_train_epochs=35,
                                   warmup_steps=15,
                                   weight_decay=0.01,    ## wd regularizor, usually a very small number as additional weight penality
                                   save_steps=30,
                                   #load_best_model_at_end=True, ## only save and load best model
                                   #metric_for_best_model='accuracy',
                                   save_total_limit = 4,        ## only save one checkpoint
                                   seed=RANDOM_SEED)  
    ## set up trainer 
    trainer = Trainer(
       model_init=model_init,
       args=training_args,
       train_dataset=input_dataset['train'],
       eval_dataset=input_dataset['val'],
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_acc# compute_acc_f1,
    )
        
    #%%
    tune_config = {
        "per_device_train_batch_size":1,
        "per_device_eval_batch_size": 1,
        "num_train_epochs": tune.choice([3, 6, 9]),
    }
    
    # Default objective is the sum of all metrics
    # when metrics are provided, so we have to maximize it.
        
    best_run = trainer.hyperparameter_search(
                hp_space=lambda _: tune_config,
                direction="maximize", 
                backend="ray", 
                resources_per_trial={"gpu": 2},
                n_trials=4 # number of trials
            )
    
    # best_trial = trainer.hyperparameter_search(
    #             direction="maximize",
    #             backend="ray",
    #             # Choose among many libraries:
    #             # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    #             search_alg=HyperOptSearch(metric="objective", mode="max"),
    #             # Choose among schedulers:
    #             # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    #             scheduler=ASHAScheduler(metric="objective", mode="max"))
        
    # ## train 
    # t = trainer.train()
    # x = trainer.evaluate() 
    # print('Training; Evaluation Results:')
    # print(t,x)
    # print('Test Results:')
    # test_res = trainer.evaluate(input_dataset['test'])
    # print(test_res)
    # #%5
    # trainer.save_model(MODEL_OUTDIR)
        
    