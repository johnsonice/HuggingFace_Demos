#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:45:23 2022

@author: chengyu
"""
import os,sys
sys.path.insert(0,'../../libs')
import config
from utils import load_jsonl,load_hp_res,get_best_hp_param,hyper_param_permutation

#%%
HP_res_dir = os.path.join(config.model_folder,'news_classification','HP_Tuning_3labels_aug_0812','hp_tuning_results_3labels.jsonl')
HP_res_processed_dir = os.path.join(config.model_folder,'news_classification','HP_Tuning_3labels_aug_0812','hp_tuning_results_3labels.csv')

tune_config_choices = {
                    'attention_probs_dropout_prob':[0.2,0.3],
                    'hidden_dropout_prob':[0.2,0.3,0.4],
                    'learning_rate':[2e-5,1e-5,5e-6,1e-6],
                    'per_device_train_batch_size':[2,4],
                    'gradient_accumulation_steps':[2,4],
                    'weight_decay':[0.05,0.1,0.2],
                    'label_smoothing_factor':[0.1,0.2,0.3],
                    'metric_for_best_model':['accuracy','loss']
                        }
## create search space
hp_space = hyper_param_permutation(tune_config_choices)

#%%
df,res=load_hp_res(HP_res_dir,sort_col=['results_eval_stats_eval_accuracy','results_test_stats_eval_accuracy'])
#%%
existing_hp = [r['hp'] for r in res]
fil_hp_space = [h for h in hp_space if h not in existing_hp]
#%%
p = get_best_hp_param(HP_res_dir,sort_col=['results_eval_stats_eval_accuracy','results_test_stats_eval_accuracy'])
#%%
