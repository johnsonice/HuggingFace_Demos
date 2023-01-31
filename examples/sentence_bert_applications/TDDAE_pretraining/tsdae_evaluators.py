#### triplet evaluator 
#%%

import os , ssl, argparse,sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
import pandas as pd 
import config 
from sentence_transformers.evaluation import TripletEvaluator
import logging
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#%%

def triplet_evaluator(input_file,n_sample=None,hard=False,batch_size=32):
    df = pd.read_excel(input_file)
    if n_sample:
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42)

    val_anchor = df['anchor_text'].to_list()
    val_positives = df['pos_text'].to_list()
    if hard:
        val_negatives = df['hardneg_text'].to_list()
    else:
        val_negatives = df['neg_text'].to_list()

    val_evaluator= TripletEvaluator(val_anchor, val_positives, val_negatives,
                                    main_distance_function=0,show_progress_bar=True,
                                    write_csv=False,batch_size=batch_size)

    return val_evaluator

def eval_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                    default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_file', action='store', dest='input_file',
                        default=os.path.join(config.data_folder,'Data/Triplet_Data/triplet_hard_neg_data.xlsx'),type=str) 
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()  
    return args

#%%

if __name__ == "__main__":
    
    args = eval_args([])
    model = SentenceTransformer(args.model_checkpoint)
    #%%
    val_evaluator = triplet_evaluator(args.input_file,n_sample=5000,hard=True)

    logging.info("Dev performance before training")
    res = val_evaluator(model)
# %%
