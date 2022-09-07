#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:47:57 2022

@author: chengyu

documentation:
    https://www.sbert.net/docs/training/overview.html

Some examples 
    https://www.pinecone.io/learn/unsupervised-training-sentence-transformers/

"""
import os 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import config
from eval import process_sts
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#%%

if __name__ == "__main__":
    
    raw_data_path = data_path = os.path.join(config.data_folder,'Data/sentence_bert/pre_training_raw_data','IMF_Documents_2018.txt')
    MODEL_OUTDIR = os.path.join(config.model_folder,'sentence_bert')
    
    raw_dataset = load_dataset('text', data_files=data_path) ## default split is 'train'
    
#%%

    # Define your sentence transformer model using CLS pooling
    model_name = 'bert-base-uncased'
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    #%%
    # Define a list with sentences (1k - 100k sentences)
    train_sentences=raw_dataset['train']['text']#[:5000]
    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
    #%%

    ## get data and setup evaluator 
    sts_samples = process_sts()
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        sts_samples, write_csv=False
    )


    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )
    
    model.save(MODEL_OUTDIR)