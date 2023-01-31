#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:47:57 2022

@author: chengyu

documentation:
    https://www.sbert.net/docs/training/overview.html

Some examples 
    https://www.pinecone.io/learn/unsupervised-training-sentence-transformers/

Possiblly use triplet evaluation data for evaluator:
    https://github.com/UKPLab/sentence-transformers/issues/1780
    https://github.com/UKPLab/sentence-transformers/issues/336
    https://github.com/UKPLab/sentence-transformers/issues/1389


"""
#%%
import os , ssl, argparse,sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
#sys.path.insert(0,'.')
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import config
from utils import get_all_files
from eval import process_sts
from tsdae_evaluators import triplet_evaluator
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def train_args(args_list=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
    #                     default=os.path.join(config.data_folder,'Models',config.default_model_checkpoint),type=str)
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                    default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN_All'),type=str) 
    parser.add_argument('--eval_file', action='store', dest='eval_file',
                        default=os.path.join(config.data_folder,'Data/Triplet_Data/triplet_hard_neg_data.xlsx'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=os.path.join(config.data_folder,'Models'),type=str)
    parser.add_argument('--model_outdir', 
                        action='store', 
                        dest='MODEL_OUTDIR',
                        default=os.path.join(config.data_folder,
                                 'Models/tsdae_pre_training_processed_{}_All'.format(config.default_model_checkpoint)),
                                type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    parser.add_argument('--vocab_aug', dest='vocab_aug',action='store_true')
    parser.add_argument('--no_vocab_aug', dest='vocab_aug',action='store_false')
    parser.set_defaults(vocab_aug=True)
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()  
    return args

#%%

if __name__ == "__main__":
    
    args = train_args([])

    MODEL = args.model_checkpoint #sentence-transformers/all-distilroberta-v1
    CACHE = args.cache_dir
    MODEL_OUTDIR= args.MODEL_OUTDIR 
    IN_DA_FOLDER = args.input_files_folder
    data_files = get_all_files(IN_DA_FOLDER,'.txt')
    assert len(data_files)>0 ## make sure we have data there 
    raw_dataset = load_dataset('text', data_files=data_files,cache_dir=CACHE)
    
    #%%
    # Define your sentence transformer model using CLS pooling
    #model_name = 'bert-base-uncased'
    word_embedding_model = models.Transformer(MODEL)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    #%%
    # Define a list with sentences (1k - 100k sentences)
    train_sentences=raw_dataset['train']['text']#[:50000]
    # Create the special denoising dataset that adds noise on-the-fly ; can not pass in customized tokenizer
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=MODEL, tie_encoder_decoder=True)
    #%%

    # ## get data and setup evaluator 
    # sts_samples = process_sts()
    # sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    #     sts_samples, write_csv=False
    # )
    funddoc_evaluator = triplet_evaluator(args.eval_file,n_sample=5000,hard=True)
    #%%

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=funddoc_evaluator,
        evaluation_steps=5000,
        epochs=1,
        weight_decay=0,
        scheduler='warmuplinear',#'constantlr',
        optimizer_params={'lr': 2e-5}, #3e-5
        #output_path = MODEL_OUTDIR,
        checkpoint_path = MODEL_OUTDIR,
        checkpoint_save_steps = 5000,
        checkpoint_save_total_limit = 5,
        show_progress_bar=True
    )
    
    model.save(MODEL_OUTDIR)
# %%
