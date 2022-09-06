#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:23:19 2022

@author: chengyu

resources:
https://huggingface.co/spaces/evaluate-measurement/perplexity

https://github.com/UKPLab/sentence-transformers/issues/1303
TODO:
    evaluate trained model on a standard task 
"""
## Model evaluation using perplexity or MLM 
import evaluate
import config
import os
#from evaluate import load

#%%
def test():
    accuracy = evaluate.load("accuracy")
    print(accuracy.description)
    print(accuracy.features)
    print(accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1]))
#%%

if __name__ == "__main_":
    raw_data_path = data_path = os.path.join(config.data_folder,'Data/sentence_bert/pre_training_raw_data','IMF_Documents_2018.txt')
    MODEL_OUTDIR = os.path.join(config.model_folder,'sentence_bert')
    #raw_dataset = load_dataset('text', data_files=data_path) 
    #%%
    ## perplexity only works fo LM models 
    perplexity = evaluate.load("perplexity", module_type="measurement")
    input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
    results = perplexity.compute(model_id=MODEL_OUTDIR,#'gpt2',
                                 add_start_token=False,
                                 data=input_texts)
    print(results)