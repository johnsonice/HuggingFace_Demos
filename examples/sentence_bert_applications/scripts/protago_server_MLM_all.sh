cd /home/chengyu.huang/project/Fund_projects/HuggingFace_Demos/examples/sentence_bert_applications/MLM_pretraining
python run_train.py --data="/home/shared_data/Language_Model_Training_Data/Data/sentence_bert/mlm_pre_training_processed_sentence-transformers/all-distilroberta-v1_All" \
--output_dir="/home/shared_data/Language_Model_Training_Data/Models/all-distillroberta-v1_adapted_All" \
--additional_vocab_path="/home/shared_data/Language_Model_Training_Data/Models/imf_vocab_aug_500.txt" \
--model_name_or_path="sentence-transformers/all-distilroberta-v1" \
--per_device_train_batch_size=32 \
--do_train \
--learning_rate=1e-4 \
--num_train_epochs=3 \
--save_step=5000 \

