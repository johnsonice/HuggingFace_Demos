#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:11:56 2022

@author: chuang
"""

#from tqdm import tqdm
from datasets import load_from_disk

# adv
#tqdm.pandas()


def get_data(args):

    dataset = load_from_disk(args.data)
    
    return dataset

