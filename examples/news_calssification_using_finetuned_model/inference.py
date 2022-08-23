#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:15:57 2022

@author: chengyu
"""

## inference 
## https://discuss.huggingface.co/t/using-trainer-at-inference-time/9378/7

import os,sys
sys.path.insert(0,'../../libs')
from transformers import Trainer,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments
from datasets import load_dataset,load_from_disk
import config
from hf_utils import compute_acc_f1
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from tqdm.auto import tqdm

def transform_pipe_results(res):   
    scores = [d['score'] for d in res]
    label = scores.index(max(scores))
    scores.append(label)
    
    return scores

def add_columns(example):
    example["text"] = '{} ; {}'.format(example['title'],example['body'])
    return example

#%%
if __name__ == "__main__":    

    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification','roberta_v2_2label_0814')
    #DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    Inference_data_dir = os.path.join(config.data_folder,'Data','climate_news','inference_data','raw','newslist_7.18.csv')
    Inference_data_out_dir = os.path.join(config.data_folder,'Data','climate_news','inference_data','infer_results_3label','newslist_7.18_pred_{}.csv')
    #%%
    ## load infer data 
    print('.....load all raw data .....')
    infer_dataset = load_dataset('csv', data_files=Inference_data_dir)
    # infer_dataset = infer_dataset.map(add_columns)
    # infer_dataset = infer_dataset.filter(lambda example:len(example['text'])>0)
    ## load pipeline
    print('.....load trained model .....')
    pipe = pipeline(task="text-classification",model = MODEL_OUTDIR,tokenizer=MODEL_OUTDIR,return_all_scores=True,device=0)
    #%%
    ##process by chunk
    chunk_size = 50000 #1000 
    n_chunk = len(infer_dataset['train'])//chunk_size
    print('chunk size : {}; number chunks : {}'.format(chunk_size,n_chunk))
    #%%
    for chunk_index in range(n_chunk):
        print('process chunk {}/{}'.format(chunk_index,n_chunk))
        chunk_dataset = infer_dataset['train'].shard(num_shards=n_chunk, index=chunk_index)
        chunk_dataset = chunk_dataset.map(add_columns)
        chunk_dataset = chunk_dataset.filter(lambda example:len(example['text'])>0)
        res = []
        for out in tqdm(pipe(KeyDataset(chunk_dataset, "text"), batch_size=8, padding=True, truncation=True)):
            res.append(transform_pipe_results(out))
        res_df = pd.DataFrame(res)
        #chunk_dataset=chunk_dataset.add_column('netural',res_df[0])
        chunk_dataset=chunk_dataset.add_column('negative',res_df[0])
        chunk_dataset=chunk_dataset.add_column('positive',res_df[1])
        chunk_dataset=chunk_dataset.add_column('label',res_df[2])
        #chunk_dataset=chunk_dataset.add_column('negative',res_df[2])
        #chunk_dataset=chunk_dataset.add_column('label',res_df[3])
        
        chunk_dataset.to_csv(Inference_data_out_dir.format(chunk_index))
        
    
    
    
    
    
    # #%%
    # test_dataset = infer_dataset['train'].shard(num_shards=n_chunk, index=0)
    # test_dataset = test_dataset.map(add_columns)
    # test_dataset = test_dataset.filter(lambda example:len(example['text'])>0)
    # #%%
    # res = []
    # for out in tqdm(pipe(KeyDataset(test_dataset, "text"), batch_size=8, padding=True, truncation=True)):
    #     res.append(transform_pipe_results(out))
    # #%%
    # res_df = pd.DataFrame(res)
    # test_dataset=test_dataset.add_column('netural',res_df[0])
    # test_dataset=test_dataset.add_column('positive',res_df[1])
    # test_dataset=test_dataset.add_column('negative',res_df[2])
    # test_dataset=test_dataset.add_column('label',res_df[3])
    # #%%
    # test_dataset.to_csv(Inference_data_out_dir.format(i))
    
    
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTDIR)
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTDIR)
    # model.eval()
    
    # #%%
    # text = ['european public goods must be paid for by the european taxpayer. the key reform as far as revenue is concerned is to grow the proportion of the autonomous â€˜own resourcesâ€™ of the eu (now only 25 per cent) and thereby to reduce direct national contributions based on gni. the eu institutions should engage with each other in developing a coherent package of new streams of revenue, including eu taxes on carbon emissions, financial transactions and aviation, as well as a distinctive tranche of vat. the new scheme must be ready to be introduced in time to fund the new medium-term multi-annual financial framework due in 2013. ']
    # encoding = tokenizer(text,return_tensors='pt')
    
    # # forward pass
    # outputs = model(**encoding)
    # predictions = outputs.logits.argmax(-1)
    # print(outputs)
    # #%%
    # input_dataset = load_from_disk(DATASET_DIR)
    # # arguments for Trainer
    # test_args = TrainingArguments(
    #     output_dir = MODEL_OUTDIR,
    #     do_train = False,
    #     do_predict = True,
    #     per_device_eval_batch_size = 4,   
    #     dataloader_drop_last = False    
    # )
    
    # # init trainer
    # trainer = Trainer(
    #               model = model, 
    #               args = test_args, 
    #               tokenizer=tokenizer,
    #               compute_metrics = compute_acc_f1)
    # #%%
    # #test_dataset=input_dataset['test'].remove_columns('text')
    # res = trainer.evaluate(input_dataset['val'])
    # print(res)
    # #%%
    # test_results = trainer.predict(input_dataset['train'])
    
    # #%%
