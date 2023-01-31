#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:31:09 2022

@author: chuang
"""

import argparse
import os, sys 
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config 
from transformers import TrainingArguments
from dataclasses import dataclass, field


def vocab_aug_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                        default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN_All'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=os.path.join(config.data_folder,'Models'),type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    args = parser.parse_args()    
    return args

def tokenize_args(args_list=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
    #                     default=os.path.join(config.data_folder,'Models',config.default_model_checkpoint),type=str)
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                    default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN_All'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=os.path.join(config.data_folder,'Models'),type=str)
    parser.add_argument('--ds_out_folder', 
                        action='store', 
                        dest='ds_out_folder',
                        default=os.path.join(config.data_folder,
                                 'Data/sentence_bert/mlm_pre_training_processed_{}_All'.format(config.default_model_checkpoint)),
                                type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    parser.add_argument('--vocab_aug', dest='vocab_aug',action='store_true')
    parser.add_argument('--no_vocab_aug', dest='vocab_aug',action='store_false')
    parser.set_defaults(vocab_aug=True)
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()  
    return args


@dataclass
class ModelTrainingArguments(TrainingArguments):
    """
    training params 
    """
    ## global pathes 
    model_name_or_path: str = os.path.join(config.data_folder,'Models',config.default_model_checkpoint)
    data: str = os.path.join(config.data_folder,
                             'Data/sentence_bert/mlm_pre_training_processed_{}_All'.format(config.default_model_checkpoint))
    output_dir: str = os.path.join(config.data_folder,'Models',config.default_model_checkpoint + '_adapted_Small')
    additional_vocab_path: str = os.path.join(config.data_folder,'Models','imf_vocab_aug_500.txt')
    
    ## global traning params  
    local_rank: int = os.getenv('LOCAL_RANK', -1) # set to run the distributed training
    dataloader_num_workers: int = 1
    n_gpu: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    #seq_length: int = 512
    total_steps: int = -1 # 125_000 #set to -1 to use total lenght of training steps  # set to control the total number of optimizer schedule steps
    max_steps: int = -1  # meant the total training steps
    learning_rate: float = 1e-4
    logging_steps: int = 500
    save_total_limit: int = 2
    save_steps: int = 5000
    evaluation_strategy: str="steps"
    eval_steps: int = 100000
    num_train_epochs: int = 5
    
    ## other traning prams 
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 1e-7
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0
    # fp16: bool = False
    # fp16_opt_level: str = "O2"
    do_train: bool = True
    do_eval: bool = True  # maybe want to set this to false if data is large 


if __name__ == '__main__':
    from transformers import HfArgumentParser
    parser = HfArgumentParser(
            ModelTrainingArguments,
    )
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

