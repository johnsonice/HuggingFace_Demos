cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--n_neighbors 5 \
--n_components 3 \
--min_cluster_size 40 \
--min_samples 10 \
--metric euclidean \
--top_n_words 5 \
--n_worker 1 \
--verbose \

