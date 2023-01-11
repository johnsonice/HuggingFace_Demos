#%%
## topic model hyper param space
import os, sys,argparse
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
from utils import hyper_param_permutation,to_jsonl,load_jsonl
from topic_arguments import topic_model_args

#%%
if __name__=="__main__":
    topic_args = topic_model_args(['--hyper_param_space_path',
                                os.path.join(config.data_folder,
                                            'Data/Raw_LM_Data/temp_topic_model/hyper_param_space.jsonl'
                                            )
                                    ]
                                )
    
    train_args = {
                'n_neighbors':[5,10,15,20,30],
                'n_components':[3,5,8],
                'min_cluster_size':[20,40,60],
                'min_samples': [1.0,0.8,0.6,0.4,0.2],
                'metric':['euclidean'],
                'top_n_words':[5,10,20],
                #'top_n_words':[5,10,15,20,30],
                #'diversity' : [0.1,0.3,0.5,0.7,0.9]
                }
    train_args_space = hyper_param_permutation(train_args)
    for t in train_args_space: ## make sure min_samples is less than min_cluster_size
        t['min_samples'] = int(t['min_cluster_size'] * t['min_samples'] )
        ## see https://github.com/MaartenGr/BERTopic/issues/582
        ## https://htmlpreview.github.io/?https://github.com/drob-xx/TopicTuner/blob/main/doc/topictuner.html#TopicModelTuner.runHDBSCAN


    to_jsonl(topic_args.hyper_param_space_path,train_args_space)

    # topic_rep_args = {
    #     'top_n_words':[5,10,15,20],
    #     'diversity' : [0.1,0.3,0.5,0.7,0.9]
    # }
    # topic_rep_args_space = hyper_param_permutation(topic_rep_args)


# %%
