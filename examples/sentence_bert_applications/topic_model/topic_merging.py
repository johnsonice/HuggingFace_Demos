
#%%
### topic merging based on manual labeled categories 
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os 
import time
import argparse
from collections import defaultdict


def model_args(args_list=None):
    MODEL_OUTPUT = '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000'
    SBERT_CHECKPOINT = '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--molder_folder', action='store', dest='MODEL_OUTPUT',
                    default=MODEL_OUTPUT,type=str)
    parser.add_argument('-tm', '--topic_model_path', action='store', dest='TOPIC_MODEL_PATH',
                    default=os.path.join(MODEL_OUTPUT,'topic_model_v2_nt1727_ch0527'),type=str)
    parser.add_argument('-tmo', '--topic_model_out_path', action='store', dest='TOPIC_MODEL_OUT_PATH',
                    default=os.path.join(MODEL_OUTPUT,'topic_model_v2_mreged_reduce_outlier'),type=str)
    parser.add_argument('-em', '--embedding_model_path', action='store', dest='SBERT_CHECKPOINT',
                    default=SBERT_CHECKPOINT,type=str)
    parser.add_argument('-ep', '--emb_path', action='store', dest='emb_path',
                    default=os.path.join(MODEL_OUTPUT,'sentence_embeddings.npy'),type=str)
    parser.add_argument('-dp', '--docs_path', action='store', dest='docs_path',
                    default=os.path.join(MODEL_OUTPUT,'docs.npy'),type=str)
    parser.add_argument('-tp', '--topics_path', action='store', dest='topics_path',
                    default=os.path.join(MODEL_OUTPUT,'topics_v2.npy'),type=str)
    parser.add_argument('-pp', '--probabiliteis_path', action='store', dest='probabiliteis_path',
                    default=os.path.join(MODEL_OUTPUT,'probabilities_v2.npy'),type=str)


    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    

    return args

def predict_test():
    t,p = topic_model.transform(docs[:10])
    print('prob dimensions:{}'.format(p[0].size))
    ## check if p is alwasy the one with highest prob
    for i in range(10):
        idx = np.array(p[i]).argmax()
        print(idx,p[i][idx])

def read_transform_topic_map(f_p):
    map_df = pd.read_csv(f_p)
    # Initialize a dictionary to store the lists
    category_dict = {}
    id2cat_dict={}
    merge_list = []
    # Group the DataFrame by 'category_id' and create lists
    for category_id, group in map_df.groupby('Cat_ID'):
        group_topic_id_list = group['Topic'].tolist()
        category_dict[category_id] = group_topic_id_list
        merge_list.append(group_topic_id_list)
        for i in group_topic_id_list:
            id2cat_dict[i] = category_id 
    
    return merge_list,category_dict,id2cat_dict


def get_eval_stats(topic_model):
    topic_df = topic_model.get_topic_info()
    topic_df['topic_words'] = topic_model.generate_topic_labels(nr_words=20,
                                                                topic_prefix=False,
                                                                word_length=100,
                                                                separator=", ")

    return topic_df

#%%
if __name__ == "__main__":
    #startTime = time.time()
    # args = topic_model_args(['--model_checkpoint',
    #                          '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000',
    #                          '--out_folder','/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000',
    #                          '--result_path','/data/chuang/Language_Model_Training_Data/Models/Topic_Models/baeline/hp_tune_results.csv'
    #                          ])

    args = model_args([])
    #%%
    ## load topic model 
    topic_model = BERTopic.load(args.TOPIC_MODEL_PATH) #,embedding_model=sentence_model
    topic_model.calculate_probabilities = True ## calculate probs for all topics 
    ## load training doc and embeddings 
    embeddings = np.load(args.emb_path)
    docs = np.load(args.docs_path)
    assert len(docs)==len(embeddings)# == len(probs) == len(topics)
    print('Number of docs: {}'.format(len(docs)))

    ## load mapping file 
    topic_map_p = os.path.join(args.MODEL_OUTPUT,'topic_map_v2.csv')   
    merge_list,category_dict,id2cat_dict=read_transform_topic_map(topic_map_p)
    old_topic_df = get_eval_stats(topic_model)
    old_topic_words_dict = old_topic_df.set_index('Topic')['topic_words'].to_dict()
    ## merge topics
    topic_model.merge_topics(docs, merge_list)
    #%%
    ## get old to new topic id map 
    old2new_topic_map = topic_model.topic_mapper_.get_mappings(original_topics=False)
    # get new to old topic map
    new2old_topic_map= defaultdict(list)
    for key, value in old2new_topic_map.items():
        new2old_topic_map[value].append(key)
    # Convert the defaultdict to a regular dictionary
    new2old_topic_map = dict(new2old_topic_map)
    newid2label = {key: id2cat_dict.get(new2old_topic_map[key][0]) for key, values in new2old_topic_map.items()}
    
    ## get old topic words merged together
    new2old_topic_words= defaultdict(list)
    for key, values in new2old_topic_map.items():
        for value in values:
            new2old_topic_words[key].append(old_topic_words_dict.get(value))
        new2old_topic_words[key] = ', '.join(new2old_topic_words[key])

    #%%
    ## update custom label 
    topic_model.set_topic_labels(newid2label)
    #%%
    topic_df = get_eval_stats(topic_model)
    # Convert the dictionary into a DataFrame
    additional_df = pd.DataFrame(new2old_topic_words.items(), columns=['Topic', 'old_topic_words'])
    topic_df = topic_df.merge(additional_df, on='Topic', how='left')
    
    #%% export model and topic info 
    topic_model.save(args.TOPIC_MODEL_OUT_PATH,save_embedding_model=True)
    topic_df.to_csv(os.path.join(args.MODEL_OUTPUT,'merged_topic_info.csv'), index=False)
    
# %%
