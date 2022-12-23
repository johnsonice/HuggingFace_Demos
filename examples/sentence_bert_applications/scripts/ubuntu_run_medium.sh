cd /home/chengyu/Dev/HuggingFace_Demos/examples/sentence_bert_applications/MLM_pretraining
python run_train.py --data="/media/chengyu/Elements1/HuggingFace/Data/sentence_bert/mlm_pre_training_processed_all-distillroberta-v1_Medium" \
--output_dir="/media/chengyu/Elements1/HuggingFace/Models/all-distillroberta-v1_adapted_Medium" \
--additional_vocab_path="/media/chengyu/Elements1/HuggingFace/Models/imf_vocab_aug_500.txt" \
--model_name_or_path="sentence-transformers/all-distilroberta-v1" \
--per_device_train_batch_size=8 \
--do_train \
--learning_rate=1e-4 \
--num_train_epochs=5 \
--save_step=5000 \
--total_steps=5250000