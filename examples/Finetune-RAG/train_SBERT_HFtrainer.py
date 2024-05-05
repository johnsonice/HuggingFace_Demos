#%%
import os , ssl, argparse,sys
sys.path.insert(0,'../../libs')
import config
from arguments import train_args
import pandas as pd
from data_utils import construct_retrieve_evaluator
from SBERT_HF_utils import SentenceTransformersCollator,SentenceTransformersTrainer
#%%
from transformers import TrainingArguments,TrainerCallback
from datasets import Dataset,load_dataset,concatenate_datasets,load_from_disk
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

import wandb

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

def load_train_eval_dataset(train_file_path=None,eval_file_path=None):
    if train_file_path:
        t_df = pd.read_excel(train_file_path)
        t_dataset = Dataset.from_pandas(t_df)
    else:
        t_dataset=None
        
    if eval_file_path:
        e_df = pd.read_excel(eval_file_path)
        e_dataset = Dataset.from_pandas(e_df)
    else:
        e_dataset=None  
          
    return t_dataset,e_dataset

# class Retrieval_Score(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         wandb.log({'retireval_score':evaluator(model)}) 


#%%
if __name__ == "__main__":
    args = train_args([])
    args.train_file = os.path.join(args.data_folder,'gpt_train_split.xlsx')
    # #args.train_file = args.eval_file
    args.eval_file = os.path.join(args.data_folder,'gpt_test_split.xlsx')
    
    text_columns=['question','context']
#%%
    #%% get training data loader
    #train_dataloader  = load_training_data_simple(args.train_file,16)
    train_ds,eval_ds = load_train_eval_dataset(args.train_file,args.eval_file)
    #%% get evaluater for retrieval
    df_eval = pd.read_excel(args.eval_file)
    evaluator = construct_retrieve_evaluator(df_eval,q_k=text_columns[0],c_k=text_columns[1])
    
    model = SentenceTransformer(args.model_checkpoint,cache_folder =args.cache_dir)
    tokenizer = model.tokenizer
    ### here we only have paired data, use multiple negative loss 
    #train_loss = losses.TripletLoss(model=model)
    loss = losses.MultipleNegativesRankingLoss(model)
    data_collator = SentenceTransformersCollator(
         tokenizer=tokenizer,
         text_columns=text_columns
    )
    def compute_metrics(eval_pred):
        return {
            "retireval_score": evaluator(model)
        }
    
    run = wandb.init(
        project="sentence_bert_HF",# Set the project where this run will be logged
        # config={
        #     "learning_rate": 0.01,
        #     "epochs": 10,
        # }
        )
    #%%
    training_arguments = TrainingArguments(
        report_to="wandb",
        output_dir=args.checkpoint_folder,
        fp16=False,
        weight_decay =0.1,
        num_train_epochs=10,
        learning_rate = 1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        logging_steps =20,
        eval_steps =80,
        save_steps = 2000,
        logging_strategy="steps",
        evaluation_strategy = "steps",
        seed=33,
        # checkpoint settings
        logging_dir=os.path.join(args.checkpoint_folder, "logs"),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="retireval_score",
        greater_is_better=True,
        # needed to get sentence_A and sentence_B
        remove_unused_columns=False,
        run_name = 'hf_test'
    )
    trainer = SentenceTransformersTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss=loss,
        text_columns=text_columns,
        compute_metrics=compute_metrics,
        #callbacks=[Retrieval_Score]
     )
    trainer.train()
    #%%
