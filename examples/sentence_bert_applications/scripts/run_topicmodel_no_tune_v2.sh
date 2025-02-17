cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--n_neighbors 5 \
--n_components 3 \
--min_cluster_size 20 \
--min_samples 20 \
--metric euclidean \
--top_n_words 5 \
--min_df 10 \
--verbose \
#--nr_topics 1500 \
#--n_worker 1 \

# n_neighbors	n_components	min_cluster_size	min_samples	top_n_words	number_topics	coherence	outlier
# 5	3	20	20	5	1782	0.501545	0.532348

# {'coherence': 0.5272108185774758, 'diversity': 0.7588882455124494, 'outlier': 0.5237587368681976, 'number_topics': 1727}
