#!/bin/bash
# Navigate to the desired directory
# https://github.com/hiyouga/LLaMA-Factory

cd /root/workspace/download/LLaMA-Factory || exit 1  

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path 'meta-llama/Meta-Llama-3-8B' \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --template default \
    --overwrite_cache \
    --dataset_dir /root/workspace/download/LLaMA-Factory/data \
    --dataset alpaca_gpt4_en \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 100 \
    --save_steps 500 \
    --warmup_steps 100 \
    --neftune_noise_alpha 0.1 \
    --optim adamw_torch \
    --packing True \
    --report_to all \
    --output_dir '/root/workspace/download/train_2024-04-21-03-22-29' \
    --fp16 True \
    --val_size 0.001 \
    --evaluation_strategy steps \
    --eval_steps 100 \
