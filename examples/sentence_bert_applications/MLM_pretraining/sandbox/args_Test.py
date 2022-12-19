#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:26:35 2022

@author: chuang
"""

### test args 
from transformers import TrainingArguments
from dataclasses import dataclass, field
import os
from transformers import HfArgumentParser


# @dataclass
# class ModelTrainingArguments(TrainingArguments):
#     model_name_or_path: str = 'roberta-base'
#     data: str = 'data/bert'
#     local_rank: int = os.getenv('LOCAL_RANK', -1) # set to run the distributed training
#     output_dir: str = "outputs"
    
#     dataloader_num_workers: int = 1
#     per_device_train_batch_size: int = 4
#     per_device_eval_batch_size: int = 4
#     gradient_accumulation_steps: int = 1
#     seq_length: int = 512

#     total_steps: int = 125_000  # set to control the total number of optimizer schedule steps
#     max_steps: int = -1  # meant the total training steps
#     learning_rate: float = 1e-4
#     warmup_steps: int = 5000
#     adam_epsilon: float = 1e-6
#     weight_decay: float = 1e-7
#     max_grad_norm: float = 1.0
#     clamp_value: float = 10000.0

    # fp16: bool = False
    # fp16_opt_level: str = "O2"
    # do_train: bool = True
    # do_eval: bool = False

    # logging_steps: int = 500
    # save_total_limit: int = 2
    # save_steps: int = 5000

    
@dataclass
class ModelTrainingArguments(TrainingArguments):
    model_name_or_path: str = 'roberta-base'
    data: str = 'data/bert'
    local_rank: int = os.getenv('LOCAL_RANK', -1) # set to run the distributed training
    output_dir: str = "outputs"
    # dataloader_num_workers: int = 1
    # per_device_train_batch_size: int = 4
    # per_device_eval_batch_size: int = 4
    # gradient_accumulation_steps: int = 1
    # #seq_length: int = 512
    # total_steps: int = 125_000  # set to control the total number of optimizer schedule steps
    # max_steps: int = -1  # meant the total training steps
    # learning_rate: float = 1e-4
    # logging_steps: int = 500
    # save_total_limit: int = 2
    # save_steps: int = 5000
    # num_train_epochs: int = 5
    
    # ## other traning prams 
    # warmup_steps: int = 5000
    # adam_epsilon: float = 1e-6
    # weight_decay: float = 1e-7
    # max_grad_norm: float = 1.0
    # clamp_value: float = 10000.0
#%%
if __name__ == '__main__':
    parser = HfArgumentParser(
            ModelTrainingArguments,
    )
    args = parser.parse_args_into_dataclasses()[0]
    print(args)