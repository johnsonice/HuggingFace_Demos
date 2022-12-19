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


def vocab_aug_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                        default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=os.path.join(config.data_folder,'Models'),type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    args = parser.parse_args()    
    return args

def tokenize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                        default=os.path.join(config.data_folder,'Models',config.default_model_checkpoint),type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN_Small'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=os.path.join(config.data_folder,'Models'),type=str)
    parser.add_argument('--ds_out_folder', 
                        action='store', 
                        dest='ds_out_folder',
                        default=os.path.join(config.data_folder,
                                 'Data/sentence_bert/mlm_pre_training_processed_{}'.format(config.default_model_checkpoint)),
                                type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    parser.add_argument('--vocab_aug', dest='vocab_aug',action='store_true')
    parser.add_argument('--no_vocab_aug', dest='vocab_aug',action='store_false')
    parser.set_defaults(vocab_aug=True)
    args = parser.parse_args() 
    return args



def training_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'roberta-base' , type=str, required=True, help='')
    parser.add_argument('--data', default= config.data_folder , type=str, required=False, help='')
    parser.add_argument('--per_device_train_batch_size', default= 16 , type=int, required=False, help='')
    parser.add_argument('--learning_rate', default= 1e-4 , type=float, required=False, help='')
    parser.add_argument('--num_train_epochs', default= 5 , type=int, required=False, help='')
    parser.add_argument('--output_dir', default= 'model_1' , type=str, required=False, help='')
    parser.add_argument('--save_epoch', default= 2 , type=int, required=False, help='')
    parser.add_argument('--save_steps', default= 5000 , type=int, required=False, help='')
    # adv
    parser.add_argument("--weight_decay", default=1e-7,type=float)
    parser.add_argument("--max_grad_norm", default=1,type=float)
    parser.add_argument("--warmup_steps", default=5000,type=float)

    # distributed learning
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="Local rank. Necessary for using the torch.distributed.launch utility")
    
  
    return parser.parse_args()

