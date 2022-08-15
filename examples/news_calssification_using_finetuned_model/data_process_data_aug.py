#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:37:48 2022

@author: chengyu

example of the process and data labeling platfrom
https://rubrix.readthedocs.io/en/stable/tutorials/01-labeling-finetuning.html

"""

### data augumentations 
import os, sys ,torch
sys.path.insert(0,'../../libs')
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset,load_from_disk,Dataset,ClassLabel,DatasetDict,concatenate_datasets
import pandas as pd
import config
from utils import flatten_list
from infer_agg_results import climate_keys
#from data_process import merge_columns,tokenize_function
from transformers import pipeline,AutoTokenizer
from inference import transform_pipe_results
    
#%%
keep_cols = ['id','pub_date', 'source', 'language', 'title', 'body', 'Year', 'ym']


def maybe_load_rawdata(sample_rawdata_dir=None,Inference_data_dir=None,initial_sample_size=500000,force_reprocess=False):
    
    if os.path.exists(sample_rawdata_dir) and not force_reprocess:
        print('....Load sample data from cache....')
        df = pd.read_pickle(sample_rawdata_dir)
        df.reset_index(drop=True,inplace=True)
    else:
        ## load infer data 
        print('.....load all raw data .....')
        infer_dataset = load_dataset('csv', data_files=Inference_data_dir)
        
        print('..... get a small sample ....')
        infer_dataset = infer_dataset.shuffle(seed=42)
        small_dataset = infer_dataset['train'].select(range(initial_sample_size))
        small_dataset.set_format("pandas")
        df = small_dataset[:]
        df['climate_count'] = df[climate_keys].sum(axis=1)
        df['climate_binary'] = df['climate_count']>0
        df = df[df['climate_binary']]
        df = df[keep_cols]
        df.reset_index(drop=True,inplace=True)
        df.to_pickle(sample_rawdata_dir)
        
    
    return df
        
        
class Search_Engine(object):
    '''
        See https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
        for doc on encode and encode multi_process
    '''
    
    def __init__(self,model_name,corpus,device=None):
        self.embedder = SentenceTransformer(model_name)
        self.corpus = corpus
        if device is None:
            ## default, SB will auto handel it, use gpu when available 
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
    def encode_corpus(self):
        self.corpus_embeddings = self.embedder.encode(corpus, 
                                                      show_progress_bar=True,
                                                      convert_to_tensor=True,
                                                      normalize_embeddings=True,
                                                      #device=self.device, 
                                                      )   ## depends on data size and gpu capacity; leave to none to let SB handle it    
        
    def search(self,query,top_k=5,return_sentence=True):
        if isinstance(query,list):
            pass
        else:
            query = list(query)
        
        query_embeddings = self.embedder.encode(queries, 
                                           convert_to_tensor=True,
                                           normalize_embeddings=True,
                                           #device=self.device,
                                           )
        
        hits = util.semantic_search(query_embeddings, 
                                    self.corpus_embeddings, 
                                    score_function=util.dot_score,
                                    top_k=top_k)
        
        if return_sentence:
            self.get_sentences(hits)
        
        return hits
    
    def get_sentences(self,hit_results):
        
        for r in hit_results:
            for i in r:
                i['sentence'] = self.corpus[i['corpus_id']]
        

def get_prediction(text,pipe):
    res = transform_pipe_results(pipe(text,padding=True,truncation=True)[0])
    return res 

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

#%%

if __name__ == "__main__":    
    
    # first, sample some data from the very large raw data set 
    Inference_data_dir = os.path.join(config.data_folder,'Data','climate_news','inference_data','raw','newslist_7.18.csv')
    data_aug_folder = os.path.join(config.data_folder,'Data','climate_news','data_augumentation')
    sample_rawdata_dir = os.path.join(data_aug_folder,'sample_rawdata.pkl')
    data_aug_dir = os.path.join(data_aug_folder,'similar_sentences.csv')
    data_aug_label_dir = os.path.join(data_aug_folder,'similar_sentences_label.csv')

    #DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset')
    DATASET_DIR = os.path.join(config.data_folder,'Data','climate_news','baseline_dataset','baseline_2label')
    MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification','roberta_v2_2label_0813')
    OUT_DATASET_DIR = os.path.join(data_aug_folder,'aug_dataset_step1_2label')
    
    model_name = "siebert/sentiment-roberta-large-english"
    
    force_resample = False
    force_repredict = False
    n_sample_to_add = 1000
    #%%
    ####
    # expand data using semetic search
    ####
    if not os.path.exists(data_aug_dir) or force_resample:
        print('..... go through data expansion process .....')
        df = maybe_load_rawdata(sample_rawdata_dir,Inference_data_dir,initial_sample_size=1000000,force_reprocess=True)
        print('loaded corpus size: {}'.format(len(df)))
        corpus = df.body.tolist()
        
        ## initiate SE object
        SE = Search_Engine('all-MiniLM-L6-v2',corpus)
        SE.encode_corpus()
        
        ## one test example 
        queries = 'the proposal would require insurers to increase capital tied to a borrower’s policy to cover potential future losses'
        res = SE.search(queries)
        print(res)
    
        ## load training queries 
        input_dataset = load_from_disk(DATASET_DIR)
        queries = input_dataset['train']['text']
        
        ## get search results and export to csv
        aug_sentences = flatten_list(SE.search(queries,top_k=20,return_sentence=False))
        #print(len(aug_sentences))
        aug_sentences = [i for i in aug_sentences if i['score']>0.65]
        #print(len(aug_sentences))
        obs_ids = [i['corpus_id'] for i in aug_sentences]
        aug_df = df.iloc[obs_ids]
        aug_df = aug_df[['title','body']]
        aug_df.to_csv(data_aug_dir,index=False)
    else:
        print('..... load pre-mapped data .....')
        aug_df = pd.read_csv(data_aug_dir)
    #%%
    ####
    # predict on expanded data 
    ####
    if not os.path.exists(data_aug_label_dir) or force_repredict:
        ### load inference pipeline 
        pipe = pipeline(task="text-classification",model = MODEL_OUTDIR,tokenizer=MODEL_OUTDIR,return_all_scores=True,device=0)
        aug_df.fillna("",inplace=True)
        aug_df["text"] = aug_df[["title", "body"]].apply(" ; ".join, axis=1)
        aug_df["text"] = aug_df["text"].apply(lambda i: i.strip(" ; "))
        ### predict on data 
        aug_df['prediction'] = aug_df['text'].apply(get_prediction,args=(pipe,))
        #aug_df[[0,1,2,'label']] = pd.DataFrame(aug_df['prediction'].tolist(), index=aug_df.index)
        aug_df[[0,1,'label']] = pd.DataFrame(aug_df['prediction'].tolist(), index=aug_df.index)
        ### filter only high confidence as label ; >0.8
        #aug_df = aug_df[(aug_df[[0,1,2]]>0.8).sum(axis=1).astype(bool)]
        aug_df = aug_df[(aug_df[[0,1]]>0.8).sum(axis=1).astype(bool)]
        aug_df.to_csv(data_aug_label_dir,index=False)
    else:
        ## load from cache
        print('load prediceted data fomr csv')
        aug_df = pd.read_csv(data_aug_label_dir)
    #%%
    ###############
    ### fromat and tokenize aug data 
    ###############
    aug_df = aug_df[['label','text']][:n_sample_to_add]
    ds = Dataset.from_pandas(aug_df)
    new_features = ds.features.copy()
    #new_features['label'] = ClassLabel(names=['Neutral','Increase Risk','Decrease Risk'])
    new_features['label'] = ClassLabel(names=['Decrease Risk','Increase Risk'])
    ds = ds.cast(new_features)
    
    ## tokenize aug data 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = ds.map(tokenize_function,batched=True,num_proc=6)
    
    ## merge with original training set 
    input_dataset = load_from_disk(DATASET_DIR)
    if input_dataset['train'].features.type == ds.features.type:
        aug_tain_ds = concatenate_datasets([input_dataset['train'],ds])
        ## replace original train data with augmented data 
        input_dataset['train'] = aug_tain_ds
    #%%
    ## export dataset 
    input_dataset.save_to_disk(OUT_DATASET_DIR)
    print('Save aug dataset to {}'.format(OUT_DATASET_DIR))