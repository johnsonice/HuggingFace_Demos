cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py \
--model_checkpoint '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000' \
--out_folder '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000' \
--n_neighbors 10 \
--n_components 8 \
--min_cluster_size 40 \
--min_samples 40 \
--metric euclidean \
--top_n_words 5 \
--verbose \


# 	n_neighbors	n_components	min_cluster_size	min_samples	top_n_words	number_topics	coherence	outlier
# 	10	8	40	40	5	383.0	0.470602	0.566043
# 	10	8	40	32	5	370.0	0.468632	0.568329

# {'coherence': 0.4646707587711406, 'diversity': 0.7763285024154589, 'outlier': 0.5891445941494375, 'number_topics': 414}
