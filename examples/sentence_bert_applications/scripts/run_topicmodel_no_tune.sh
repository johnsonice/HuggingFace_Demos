cd /home/chuang/Dev/HuggingFace_Demos/examples/sentence_bert_applications/topic_model
python bert_topic_hyper_tuning.py --n_neighbors 30 \
--n_components 8 \
--min_cluster_size 60 \
--min_samples 48 \
--metric euclidean \
--top_n_words 20 \
--n_worker 1 \
--verbose \

