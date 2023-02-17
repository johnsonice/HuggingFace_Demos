cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--tune \
--n_worker 20 \
--chunk_size 200 \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--result_path '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000/hp_tune_results.csv' \
#--verbose \
#--test_run \

