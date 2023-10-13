#### most basic training process 
#%%
import os , ssl, argparse,sys
sys.path.insert(0,'../../libs')
import config
from arguments import train_args
import pandas as pd
from data_utils import construct_retrieve_evaluator

from datasets import Dataset,load_dataset,concatenate_datasets,load_from_disk
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
import wandb

def callback_wb(score,epoch,steps):
    if steps <= 0:
        pass
    else:
        steps = epoch*len(train_dataloader) + steps
        wandb.log({'acc':score,'epoch':epoch,'steps':steps})
    
    return None

def load_training_data_simple(train_file_path,batch_size=4):
    """
    train_file_path : training data excel file path
    """
    df = pd.read_excel(train_file_path)
    df = df[['question','context']]
    #df.columns= ['query','context']
    examples = []
    for index,row in df.iterrows():
        example = InputExample(texts=[row['question'],row['context']])
        examples.append(example)
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size) 
    
    return train_dataloader
    #%%

if __name__ == "__main__":
    args = train_args([])
    # args.train_file = os.path.join(args.data_folder,'human_train_split.xlsx')
    # #args.train_file = args.eval_file
    # args.eval_file = os.path.join(args.data_folder,'human_test_split.xlsx')
    #%% get training data loader
    train_dataloader  = load_training_data_simple(args.train_file,16)
    
    #%% get evaluater for retrieval
    df_eval = pd.read_excel(args.eval_file)
    evaluator = construct_retrieve_evaluator(df_eval,q_k='question',c_k='context')
    
    #%%
    model = SentenceTransformer(args.model_checkpoint,cache_folder =args.cache_dir)
    ### here we only have paired data, use multiple negative loss 
    #train_loss = losses.TripletLoss(model=model)
    loss = losses.MultipleNegativesRankingLoss(model)
    #%%
    EPOCHS = 20
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
    run = wandb.init(
        project="sentence_bert_test",# Set the project where this run will be logged
        # config={
        #     "learning_rate": 0.01,
        #     "epochs": 10,
        # }
        )

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        scheduler='warmuplinear',#'constantlr',
        optimizer_params={'lr': 1e-5}, #3e-5
        output_path=args.checkpoint_folder,
        show_progress_bar=True,
        evaluator=evaluator, 
        evaluation_steps=20,
        checkpoint_save_steps = 2000,
        checkpoint_save_total_limit = 5,
        callback=callback_wb
    )


















