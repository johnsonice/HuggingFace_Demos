#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:38:03 2022

@author: chengyu
## mainly follow:
    https://huggingface.co/course/chapter3/2?fw=pt
    https://medium.com/mlearning-ai/twitter-sentiment-analysis-with-deep-learning-using-bert-and-hugging-face-830005bcdbbf
    https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
    https://lewtun.github.io/blog/til/nlp/huggingface/transformers/2021/01/01/til-data-collator.html
    https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
    https://huggingface.co/blog/sentiment-analysis-python
## weight decay:
    https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab

## dynamic padding with data collector:
    https://huggingface.co/course/chapter3/2?fw=pt
    
## for training logs:
    https://docs.wandb.ai/quickstart    
    
"""

import sys,os
sys.path.insert(0, '../../libs')
from utils import load_jsonl
import config
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

## import transformer packages 
import datasets
from datasets import Dataset,load_metric,logging as dataset_logging
from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,logging as transformers_logging
from transformers import TrainingArguments, Trainer
transformers_logging.set_verbosity_error() ## make it less verbose
dataset_logging.set_verbosity_error() ## make it less verbose
print('gpu available : {}'.format(torch.cuda.is_available()))

#%%
def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels,average=None)["f1"] 
   if isinstance(f1, np.ndarray):
       f1 = f1.tolist()  ## add this as nd array is not serializable when exporting 
   # print(load_metric("f1").inputs_description) ## print out descriptions 
   return {"accuracy": accuracy, "f1": f1}


#%%
if __name__ == "__main__":
    
    ## set up global params 
    RANDOM_SEED = config.RANDOM_SEED
    MODEL = "bert-base-cased"
    N_CPU = 6 
    out_dir = os.path.join(config.data_folder,'tweet_classfier')
    #%%
    ## load csv data into df
    data_path = os.path.join(config.data_folder,'tweets.csv')
    df = pd.read_csv(data_path,encoding='utf8')
    df = df[~df['text'].isna()]  ## clear NaN, otherwise dataset function will complain
    df['label']= df['label'].astype(int)
    #%%
    ## prepare dataset
    df_train, df_test = train_test_split(df,test_size=0.3,random_state=RANDOM_SEED)
    dataset_train = Dataset.from_pandas(df_train,split='train')
    dataset_test = Dataset.from_pandas(df_test,split='test')
    dataset = datasets.DatasetDict({'train':dataset_train, 'test':dataset_test})
    print(dataset)
    
    #%%
    ## process dataset / tokenize and encode / tokenizer should change based on model selection 
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True) #padding="max_length", ; normally you want to pad to global max length
                                                            # here we will use data collector later which will automately pad based on 
                                                            # longest of the batch 
                                                            # for global max padding see references on the top 
                                                            
    ## the will process both train and test split 
    tokenized_datasets = dataset.map(tokenize_function, batched=True,num_proc=N_CPU) ##batch proceee then ; and multiprocess = 6 
    #print(tokenized_datasets)
    ## remove unused columns 
    print(tokenized_datasets['train'][0])
        ## dict_keys(['text', 'label', '__index_level_0__'])
    tokenized_datasets= tokenized_datasets.remove_columns(['text','__index_level_0__'],)
    ## split out train and test 
    tokenized_train = tokenized_datasets['train']
    tokenized_test = tokenized_datasets['test']
    
    #%%
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    ## set up training arguments 
    #%%
    training_args = TrainingArguments(output_dir=out_dir,
                                       #evaluation_strategy="epoch",
                                       evaluation_strategy="steps",
                                       eval_steps=20,
                                       logging_steps =20,          ## show eval results
                                       learning_rate=2e-5,
                                       per_device_train_batch_size=8,
                                       per_device_eval_batch_size=8,
                                       num_train_epochs=20,
                                       weight_decay=0.4, ## wd regularizor, usually a very small number as additional weight penality
                                       save_steps=100,
                                       load_best_model_at_end=True, ## only save and load best model
                                       save_total_limit = 1,        ## only save one checkpoint
                                       seed=RANDOM_SEED)  
    ## set up trainer 
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )
        
    ## train 
    t = trainer.train()
    x = trainer.evaluate() 
    print(t,x)
    trainer.save_model(out_dir)
    
    
    
    