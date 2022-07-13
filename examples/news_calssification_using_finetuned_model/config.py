#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:40:55 2022

@author: chengyu
"""
import os 

## some global params 
data_folder1= "/media/chengyu/Elements1/HuggingFace"
data_folder2= "/Users/huang/Dev/projects/All_Data/HuggingFace"
if os.path.exists(data_folder1):
    data_folder = data_folder1
elif os.path.exists(data_folder2):
    data_folder = data_folder2
RANDOM_SEED = 42