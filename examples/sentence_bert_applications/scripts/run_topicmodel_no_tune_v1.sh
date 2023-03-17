cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--n_neighbors 5 \
--n_components 3 \
--min_cluster_size 20 \
--min_samples 16 \
--metric euclidean \
--top_n_words 5 \
--verbose \


# 	n_neighbors	n_components	min_cluster_size	min_samples	top_n_words	number_topics	coherence	outlier
# 	5	3	20	16	5	2119.0	0.513490	0.535554

# {'coherence': 0.5266519593403612, 'diversity': 0.7545323741007194, 'outlier': 0.5379187418257486, 'number_topics': 2085}
