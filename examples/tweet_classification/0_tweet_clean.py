# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:34:56 2022

@author: CHuang
"""

#import numpy as np
import re,os
import pandas as pd
import config 
 
def clean_tweet(tweet,stopwords=[]):
    if type(tweet) == float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

def simple_clean_tweet(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

#%%
if __name__ == "__main__":
    file_name = 'sample_tweets.csv'
    raw_file = os.path.join(config.data_foder,file_name)
    
    df = pd.read_csv(raw_file,encoding='latin1') ## some encoding issues
    df.columns.values[:4] = ['1','metadata','text','label']
    df = df[['text','label']]
    ## in our data it is {0:'Negative', 1:"Positive",2:"Nutural"}
    df['label'].fillna(2,inplace=True)
    df['text'] = df['text'].map(clean_tweet)
    
    ## export data 
    df.to_json(os.path.join(config.data_foder,'tweets.jsonl'),orient='records', lines=True)
    df.to_csv(os.path.join(config.data_foder,'tweets.csv'),encoding='utf8',index=False)
    # ## split and export if you need to preprocess them 
    # df_train = df.sample(frac = 0.25,replace=False)
    # df_test = df.drop(df_train.index)  ## drop already sampled obs 
    # df_train.to_json(os.path.join(config.data_foder,'tweets_train.jsonl'),orient='records', lines=True)
    # df_test.to_json(os.path.join(config.data_foder,'tweets_test.jsonl'),orient='records', lines=True)

    
    
    