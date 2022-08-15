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
import gc,torch
from utils import hyper_param_permutation,to_jsonl,load_hp_res
import copy
## modeify the head of someone's fintuned model : use the ignore mismatched size argument
## https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/19 

## hyperparameter-search in trainer
# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785?page=3
# https://huggingface.co/blog/ray-tune
#%%

def clear_memory(verbose=False):
    #stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.') 

def train_one_trail(model_name,input_dataset,data_collator,hy_config,n_label=2):
    tune_config = copy.copy(hy_config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=n_label,
                                                               ignore_mismatched_sizes=True,
                                                               attention_probs_dropout_prob=tune_config['attention_probs_dropout_prob'], ##change drop out 
                                                               hidden_dropout_prob=tune_config['hidden_dropout_prob']) ##change drop out 
    
    tune_config.pop('attention_probs_dropout_prob')
    tune_config.pop('hidden_dropout_prob')
    ## set up training args 
    training_args = TrainingArguments(output_dir=MODEL_OUTDIR,
                                   #evaluation_strategy="epoch",
                                   evaluation_strategy="steps",
                                   eval_steps=20,
                                   logging_steps =20,          ## show eval results
                                   per_device_eval_batch_size=2,                              
                                   num_train_epochs=5,
                                   warmup_steps=5,
                                   save_steps=20,
                                   load_best_model_at_end=True, ## only save and load best model
                                   save_total_limit = 1,        ## only save one checkpoint
                                   seed=RANDOM_SEED,
                                   **tune_config
                                   #gradient_accumulation_steps=2,
                                   #learning_rate=1e-5,
                                   #per_device_train_batch_size=4,
                                   #weight_decay=0.01,    ## wd regularizor, usually a very small number as additional weight penality
                                   #label_smoothing_factor=0.1,
                                   #metric_for_best_model='accuracy',
                                   )  
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
    
    ## train 
    train_stats = trainer.train()
    eval_stats = trainer.evaluate() 
    test_stats = trainer.evaluate(input_dataset['test'])
    
    return {'train_stats':train_stats,'eval_stats':eval_stats,'test_stats':test_stats}

if __name__ == "__main__":
    
    N_LABLE = 2 # previous it was 3 
    N_CPU = config.N_CPU 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification')
    #DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    HP_RES_DIR = os.path.join(MODEL_OUTDIR,'hp_tuning_results.jsonl')
    
    #DS_version = 'data_augumentation/aug_dataset_step1'#baseline_dataset
    DS_version = 'data_augumentation/aug_dataset_step1_2label'#baseline_2label'#data_augumentation/aug_dataset_step1'#baseline_dataset
    run_name='climate_news/'+DS_version
    DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news',DS_version)
    
    tune_config_choices = {
                        'attention_probs_dropout_prob':[0.2,0.3],
                        'hidden_dropout_prob':[0.2,0.3],
                        'learning_rate':[2e-5,1e-5,5e-6,1e-6],
                        'per_device_train_batch_size':[2],
                        'gradient_accumulation_steps':[2,4],
                        'weight_decay':[0.05,0.1,0.2],
                        'label_smoothing_factor':[0.1,0.2,0.3],
                        'metric_for_best_model':['accuracy','loss'],
                        'run_name':[run_name]
                            }
    ## create search space
    hp_space = hyper_param_permutation(tune_config_choices)

    if os.path.exists(HP_RES_DIR):
        chpl = len(hp_space)
        df,res=load_hp_res(HP_RES_DIR,sort_col=['results_eval_stats_eval_accuracy','results_test_stats_eval_accuracy'])
        existing_hp = [r['hp'] for r in res]
        hp_space = [h for h in hp_space if h not in existing_hp]
        chpl2 = len(hp_space)
        print('Read and continue with existing HP results, reduce purmutatio from {} to {}'.format(chpl,chpl2))
    
    #%%
    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## load pretrained and fintuned model with a ramdom initialized classifer layer
    ## load an sample dataset 
    input_dataset = load_from_disk(DATASET_DIR)
    print(input_dataset)
    input_dataset = input_dataset.remove_columns('text') ## remove raw text for training
    ## set up data collactor for dynamic padding in batch 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    for hp in hp_space:
        res = train_one_trail(model_name,input_dataset,data_collator,hp,n_label = N_LABLE)
        print('train eval results:')
        print(hp,res['train_stats'],res['eval_stats'])
        to_jsonl(HP_RES_DIR,{'hp':hp,'results':res},mode='a')
        del res
        clear_memory()

    