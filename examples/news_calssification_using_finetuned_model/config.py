#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:40:55 2022

@author: chengyu
"""
import os 

## global folder path 
data_folder1= "/media/chengyu/Elements1/HuggingFace"
data_folder2= "/Users/huang/Dev/projects/All_Data/HuggingFace"

if os.path.exists(data_folder1):
    data_folder = data_folder1  ## for linux
elif os.path.exists(data_folder2):
    data_folder = data_folder2  ## for mac

model_folder = os.path.join(data_folder,'Models')    



## other params 
RANDOM_SEED = 42
N_CPU = 6