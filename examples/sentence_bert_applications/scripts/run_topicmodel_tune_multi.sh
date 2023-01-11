cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--tune \
--n_worker 10 \
--chunk_size 40 \
--result_path "/data/chuang/Language_Model_Training_Data/Data/Raw_LM_Data/temp_topic_model/hp_tune_results.csv"
#--test_run \

