#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:38:03 2022

@author: chengyu
## mainly follow:
    https://medium.com/mlearning-ai/twitter-sentiment-analysis-with-deep-learning-using-bert-and-hugging-face-830005bcdbbf
    
"""

import sys,os
sys.path.insert(0, '../../libs')
from utils import load_jsonl
from datasets import load_dataset
import torch
#%%
