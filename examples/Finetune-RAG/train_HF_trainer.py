### train customized

"""
Follow : 
- https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/train_script.py
- https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/modeling.py
"""
#%%
import sys,os
sys.path.insert(0,'../../libs')
from arguments import train_args
from data_utils import load_train_eval_dataset
from SBERT_HF_utils import SentenceTransformersCollator
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from models import AutoModelForSentenceEmbedding

class BiTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

#%%
if __name__ == "__main__":
    args = train_args([])
    train_ds,eval_ds = load_train_eval_dataset(args.train_file,args.eval_file)
    tokenizer =  AutoTokenizer.from_pretrained(args.model_checkpoint,cache_folder =args.cache_dir)
    ## let's just tokenize it on the fly using data collator
    text_columns = ['question','context']
    data_collator = SentenceTransformersCollator(
         tokenizer=tokenizer,
         text_columns=text_columns
    )
    #batch_size = 8
    #train_loader = DataLoader(train_ds, batch_size=batch_size,collate_fn=data_collator)
    #eval_loader = DataLoader(eval_ds, batch_size=batch_size,collate_fn=data_collator)
    # %%
    model = AutoModelForSentenceEmbedding(args.model_checkpoint,cache_dir=args.cache_dir)

#%%
    # setup warmup for first ~10% of steps
    # epochs=3
    # total_steps = int(len(train_ds)*epochs / batch_size)
    # warmup_steps = int(0.1 * total_steps)


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
        gradient_accumulation_steps=2,
        logging_steps =20,
        eval_steps =40,
        save_steps = 2000,
        logging_strategy="steps",
        evaluation_strategy = "steps",
        seed=33,
        # checkpoint settings
        logging_dir=os.path.join(args.checkpoint_folder, "logs"),
        save_total_limit=2,
        load_best_model_at_end=True,
        #metric_for_best_model="retireval_score",
        greater_is_better=True,
        # needed to get sentence_A and sentence_B
        remove_unused_columns=False,
        run_name = 'hf_test_2'
    )
    trainer = BiTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        #loss=loss,
        #text_columns=text_columns,
        #compute_metrics=compute_metrics,
        #callbacks=[Retrieval_Score]
     )
    trainer.train()

