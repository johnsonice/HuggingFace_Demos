#%%
### Topic model training arguments 
import os, sys,argparse
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
from utils import hyper_param_permutation


def topic_model_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                    default='all-distilroberta-v1',type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--input_files_folder', action='store', dest='input_files_folder',
                        default=os.path.join(config.data_folder,'Data/Raw_LM_Data/CLEAN_Small'),type=str) 
    parser.add_argument('--out_folder', action='store', dest='out_folder',
                        default=os.path.join(config.data_folder,'Models/Topic_Models/baeline'),type=str)
    parser.add_argument('--result_path', action='store', dest='result_path',
                        default=os.path.join(config.data_folder,'Models/Topic_Models/baeline/hp_tune_results.csv'),type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.data_folder,'cache'),type=str) 
    parser.add_argument('--hyper_param_space_path', action='store', dest='hyper_param_space_path',
                        default=os.path.join(config.data_folder,'Models/Topic_Models/hyper_param_space.jsonl'),type=str)                                  
    parser.add_argument('--n_neighbors', action='store', dest='n_neighbors',
                            default=15,type=int) 
    parser.add_argument('--n_components', action='store', dest='n_components',
                            default=5,type=int) 
    parser.add_argument('--min_cluster_size', action='store', dest='min_cluster_size',
                            default=20,type=int) 
    parser.add_argument('--min_samples', action='store', dest='min_samples',
                            default=10,type=int) 
    parser.add_argument('--top_n_words', action='store', dest='top_n_words',
                            default=20,type=int)  ## this does affect topic reduction process and thus topic identification 
    parser.add_argument('--diversity', action='store', dest='diversity',
                            default=0.8,type=float) 
    parser.add_argument('--metric', action='store', dest='metric',
                            default='euclidean',type=str) 
    parser.add_argument('--min_df', action='store', dest='min_df',
                            default=10,type=str) 
    parser.add_argument('--n_worker', action='store', dest='n_worker',
                            default=1,type=int)                  
    parser.add_argument('--chunk_size', action='store', dest='chunk_size',
                            default=20,type=int) 
    parser.add_argument('--no_load_emb', action='store_false', dest='LOAD_EMB') 
    parser.add_argument('--tune', action='store_true', dest='TUNE') 
    parser.add_argument('--cal_prob', action='store_true', dest='calculate_probabilities') 
    parser.add_argument('--verbose', action='store_true', dest='verbose') 
    parser.add_argument('--test_run', action='store_true', dest='test_run') 

    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    
    return args

# train_args = {
#             'n_neighbors':[5,10,15,20,25,30],
#             'n_components':[3,5,8,10],
#             'min_cluster_size':[20,40,60,80],
#             'min_samples': [1.0,0.8,0.6,0.4,0.2],
#             'metric':['euclidean'],
#             'top_n_words':[5,10,20,30],
#             #'top_n_words':[5,10,15,20,30],
#             #'diversity' : [0.1,0.3,0.5,0.7,0.9]
#             }
# train_args_space = hyper_param_permutation(train_args)
# for t in train_args_space: ## make sure min_samples is less than min_cluster_size
#     t['min_samples'] = int(t['min_cluster_size'] * t['min_samples'] )
#     ## see https://github.com/MaartenGr/BERTopic/issues/582
#     ## https://htmlpreview.github.io/?https://github.com/drob-xx/TopicTuner/blob/main/doc/topictuner.html#TopicModelTuner.runHDBSCAN

# topic_rep_args = {
#     'top_n_words':[5,10,15,20],
#     'diversity' : [0.1,0.3,0.5,0.7,0.9]
# }
# topic_rep_args_space = hyper_param_permutation(topic_rep_args)

#%%
if __name__ == "__main__":
    t_args = topic_model_args()
    print(t_args)

    # hyper_params_list =  hyper_param_permutation(train_args)
    # arg_v1 = hyper_params_list[0]
    # print(arg_v1)
    # t_args.__dict__.update(arg_v1)
    # print(t_args)
    #%%
#args_list = 
