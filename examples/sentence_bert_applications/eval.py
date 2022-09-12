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
    
    
SBERT sequential evaluator :
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/SequentialEvaluator.py
    https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.SequentialEvaluator
    https://github.com/UKPLab/sentence-transformers/issues/458
    give it a try 
"""
#%%
## Model evaluation using perplexity or MLM 
import evaluate
#import config
import os
from datasets import load_dataset,load_from_disk
from sentence_transformers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models
#%%
def test():
    accuracy = evaluate.load("accuracy")
    print(accuracy.description)
    print(accuracy.features)
    print(accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1]))

def eval_LM_model(model_path,input_texts):
    ## perplexity only works fo LM models 
    perplexity = evaluate.load("perplexity", module_type="measurement")
    
    results = perplexity.compute(model_id=model_path,#'gpt2',
                                 add_start_token=False,
                                 data=input_texts)
    print(results)

def process_sts():
    sts = load_dataset('glue', 'stsb', split='validation')
    # normalize the 0 -> 5 range
    sts = sts.map(lambda x: {'label': x['label'] / 5.0})

    samples = []
    for sample in sts:
        # reformat to use InputExample
        samples.append(InputExample(
            texts=[sample['sentence1'], sample['sentence2']],
            label=sample['label']
        ))
    return samples 

#%%
def eval_pretrained_embeding_models(eval_samples,model_name='bert-base-uncased'):
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_samples, write_csv=False
    )
    bert = models.Transformer(model_name)
    pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[bert, pooling])
    return evaluator(model)

def eval_sentencebert_models(eval_samples,model_name='bert-base-nli-mean-tokens'):
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_samples, write_csv=False
    )
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    return evaluator(model)

#%%

if __name__ == "__main__":
    
    #######
    ## use huggingface to evaluate LM model with perplexity score
    #######
    
    #raw_data_path = data_path = os.path.join(config.data_folder,'Data/sentence_bert/pre_training_raw_data','IMF_Documents_2018.txt')
    #MODEL_OUTDIR = os.path.join(config.model_folder,'sentence_bert')
    MODEL_OUTDIR = 'gpt2'
    #raw_dataset = load_dataset('text', data_files=data_path) 
    input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
    #eval_LM_model(MODEL_OUTDIR,input_texts)
    
    #######
    ## use sbert to evaluate any pretrained embeding model for paragraph similarity task
    #######
    sts_eval_samples=  process_sts()
    res_score = eval_pretrained_embeding_models(sts_eval_samples,model_name='bert-base-uncased')
    print('pretrained model similarity score : {}'.format(res_score))

    ######################
    ## use sbert to evaluate sentencebert model for paragraph similarity task
    ######################
    sts_eval_samples=  process_sts()
    res_score = eval_sentencebert_models(sts_eval_samples,model_name='bert-base-nli-mean-tokens')
    print('pretrained sentence bert mode similarity score : {}'.format(res_score))




