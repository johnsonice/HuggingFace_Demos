cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model/sandbox
python bert_topic_hyper_tuning_benchmark.py \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--tune \
--n_worker 20 \
--chunk_size 50 \
--n_neighbors 10 \
--n_components 8 \
--min_cluster_size 20 \
--min_samples 16 \
--top_n_words 30 \
--metric euclidean \
--result_path "/data/chuang/Language_Model_Training_Data/Data/Raw_LM_Data/temp_topic_model/hp_tune_results_benchmark.csv" \
--test_run \
#--verbose


