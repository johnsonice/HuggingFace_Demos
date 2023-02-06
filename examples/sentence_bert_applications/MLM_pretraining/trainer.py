#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:10:03 2022

@author: chuang
"""


from transformers.trainer import Trainer
from transformers import  DefaultFlowCallback #ProgressCallback,
#import torch
import os
from tqdm import tqdm
import logging
# adv
#from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup #get_cosine_schedule_with_warmup,
from pathlib import Path
tqdm.pandas()
logger = logging.getLogger(__name__)

def train(tokenizer, data_collator, tokenized_datasets, model, optimizer, args):
    print('start training')

    # set up schedule if needed
    # linear warmup
    #num_trianing_steps = args.num_train_epochs * int(len(tokenized_datasets['train'])/args.n_gpu/args.per_device_train_batch_size)
    if args.total_steps > 0:
        schedule_total = args.total_steps
    else:
        schedule_total = args.num_train_epochs * int(len(tokenized_datasets['train'])/args.n_gpu/args.per_device_train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=schedule_total
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'] if args.do_train else None,
        eval_dataset=tokenized_datasets['test'] if args.do_eval else None,
        optimizers=(optimizer, scheduler),
        callbacks=[DefaultFlowCallback],
        ###WanB related arguments
        # report_to="wandb",
        # run_name=model_name
    )
    #trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    #trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    
    # Training
    if args.do_train:
        latest_checkpoint_dir = max(
            Path(args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime
        )

        trainer.train(resume_from_checkpoint=latest_checkpoint_dir)
        
    return trainer