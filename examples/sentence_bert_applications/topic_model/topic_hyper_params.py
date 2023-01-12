#%%
## topic model hyper param space
import os, sys,argparse
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
from utils import hyper_param_permutation,to_jsonl,load_jsonl
from topic_arguments import topic_model_args
import pandas as pd 

#%%
def load_hyper_params(f_p):
    if '.jsonl' in f_p:
        params_space = load_jsonl(f_p)
        return params_space
    else:
        raise Exception('hyper params must be in jsonl format')

def get_params_diff(p_space_fp,tune_res_fp=None):
    """load and compare finished results with new param space"""
    
    params_space = load_hyper_params(p_space_fp)
    if tune_res_fp and os.path.isfile(tune_res_fp):
        tune_res_df = pd.read_csv(tune_res_fp)
        res_param_space = tune_res_df[list(params_space[0].keys())].to_dict('records')
        dif_params_space = [p for p in params_space if p not in res_param_space]
        return dif_params_space
    else:
        print('No previously calculated results, use entire params space for tuning.')
        return params_space
    
#%%
if __name__=="__main__":
    topic_args = topic_model_args(['--hyper_param_space_path',
                                os.path.join(config.data_folder,
                                            'Data/Raw_LM_Data/temp_topic_model/hyper_param_space.jsonl'
                                            )
                                    ]
                                )
    
    train_args = {
                'n_neighbors':[5,10,15,20,25,30],
                'n_components':[3,5,8,10],
                'min_cluster_size':[20,40,60,80],
                'min_samples': [1.0,0.8,0.6,0.4,0.2],
                'metric':['euclidean'],
                'top_n_words':[5,10,20,30],
                #'top_n_words':[5,10,15,20,30],
                #'diversity' : [0.1,0.3,0.5,0.7,0.9]
                }
    train_args_space = hyper_param_permutation(train_args)
    for t in train_args_space: ## make sure min_samples is less than min_cluster_size
        t['min_samples'] = int(t['min_cluster_size'] * t['min_samples'] )
        ## see https://github.com/MaartenGr/BERTopic/issues/582
        ## https://htmlpreview.github.io/?https://github.com/drob-xx/TopicTuner/blob/main/doc/topictuner.html#TopicModelTuner.runHDBSCAN

    #%%
    to_jsonl(topic_args.hyper_param_space_path,train_args_space)
    add_params = get_params_diff(topic_args.hyper_param_space_path,topic_args.result_path)
    print("additional params to try :{}".format(len(add_params)))


# %%
