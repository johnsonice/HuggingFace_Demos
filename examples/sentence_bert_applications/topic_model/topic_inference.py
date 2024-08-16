#%%
### topic merging based on manual labeled categories 
from bertopic import BERTopic
import pandas as pd
import os 
import time
import numpy as np
import argparse
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed



def model_args(args_list=None):
    MODEL_OUTPUT = '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000'
    SBERT_CHECKPOINT = '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--molder_folder', action='store', dest='MODEL_OUTPUT',
                    default=MODEL_OUTPUT,type=str)
    parser.add_argument('-tmo', '--topic_model_out_path', action='store', dest='TOPIC_MODEL_OUT_PATH',
                    default=os.path.join(MODEL_OUTPUT,'topic_model_v2_mreged'),type=str)
    parser.add_argument('-dp', '--docs_path', action='store', dest='docs_path',
                    default=os.path.join(MODEL_OUTPUT,'docs.npy'),type=str)
    parser.add_argument('-tp', '--topics_path', action='store', dest='topics_path',
                    default=os.path.join(MODEL_OUTPUT,'topics_v2.npy'),type=str)
    parser.add_argument('-txf', '--text_folder', action='store', dest='text_folder',
                    default=os.path.join(MODEL_OUTPUT,'text_data'),type=str)
    parser.add_argument('-txfn', '--text_filename', action='store', dest='text_filename',
                    default='program.csv',type=str)
    parser.add_argument('-cache', '--cache_folder', action='store', dest='cache_folder',
                    default=os.path.join(MODEL_OUTPUT,'results_cache'),type=str)
    parser.add_argument('-batch_id_range', '--batch_id_range', action='store', dest='batch_id_range',
                        default="0-10000",type=str)
    
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    

    return args

def batch_array(arr, batch_s):
    """
    Batch an array into smaller batches of a specified size.
    Args:
    arr (list or numpy.ndarray): The input array to be batched.
    batch_size (int): The size of each batch.
    Returns:
    list: A list of batches, where each batch is a sublist of the original array.
    """
    batched_array = []
    for i in range(0, len(arr), batch_s):
        batched_array.append(arr[i:i + batch_s])
    return batched_array

class TM(object):
    def __init__(self, model_path):
        self.model = BERTopic.load(model_path)
        self.model.calculate_probabilities = True 

    def predict_topic(self,docs):
        t,p = self.model.transform(docs)
        #print('prob dimensions:{}'.format(p[0].size))
        ## check if p is alwasy the one with highest prob
        return t,p
    
    def predict_no_outlier(self,input_test_list):
        org_topics,p = self.model.transform(input_test_list)
        print('... reducing outlier ...')
        new_topics = self.model.reduce_outliers(input_test_list, org_topics,strategy="embeddings")
        return new_topics,p
    
    def predict_and_merge(self,df,doc_col_name,reduce_outlier=True):
        print('.... tranforming docs ....')
        if reduce_outlier:
            t,p = self.predict_no_outlier(df[doc_col_name])
        else:
            t,p = self.model.transform(df[doc_col_name])
        p_max_id = np.argmax(p,axis=1)
        max_prob = p[np.arange(len(p_max_id)), p_max_id]
        prob_df = pd.DataFrame(p)
        t_data = {'org_topic_id': t, 'prob_topic_id': p_max_id,'max_probability':max_prob}
        topic_df = pd.DataFrame(t_data)
        # Concatenate DataFrames in sequence
        final_df = pd.concat([df, topic_df, prob_df], axis=1,ignore_index=True)
        final_df.columns =list(df.columns)+list(topic_df.columns)+list(prob_df.columns)

        return final_df

    def batch_predict_topic(self,doc_df,doc_col_name,batch_size=200,
                            accumulate=True,cache_dir=None,batch_id_range=None):
        
        doc_batches = batch_array(doc_df,batch_size)
        res_list = []
        if batch_id_range:
            try:
                s,e = batch_id_range.split('-')
                s = int(s)
                if len(e)>0:
                    e = int(e)
                else:
                    e = int(10e10)
                batch_id_range = range(s,e)
                print('Working for batches {} - {}'.format(s,e))
            except:
                batch_id_range=None
                raise('parsing batch_id range failed')
        
        for idx,db in tqdm(enumerate(doc_batches),total = len(doc_batches)):
            #print(db[doc_col_name])
            #print('batch:{}'.format(idx))
            if batch_id_range:
                if idx in batch_id_range:
                    print('working on batch : {}/{} ....'.format(idx,len(doc_batches)))
                    db=db.reset_index()
                    final_df = self.predict_and_merge(db,doc_col_name)
                    if accumulate:
                        res_list.append(final_df)
                    if cache_dir:
                        final_df.to_csv(os.path.join(cache_dir,'batch_{}.csv'.format(idx)),index=False)
                else:
                    continue
            else:
                print('working on batch : {}/{} ....'.format(idx,len(doc_batches)))
                db=db.reset_index()
                final_df = self.predict_and_merge(db,doc_col_name)
                if accumulate:
                    res_list.append(final_df)
                if cache_dir:
                    final_df.to_csv(os.path.join(cache_dir,'batch_{}.csv'.format(idx)),index=False)
        
        if len(res_list)>0:
            all_df = pd.concat(res_list, axis=0, ignore_index=True)
            if cache_dir:
                if batch_id_range:
                    all_df.to_csv(os.path.join(cache_dir,'batch_{}.csv'.format(batch_id_range)),index=False)
                else:
                    all_df.to_csv(os.path.join(cache_dir,'batch_{}.csv'.format(all)),index=False)

            return all_df
        return res_list

#%%

if __name__ == "__main__":
    #startTime = time.time()
    args = model_args([])
    #%%
    fp = os.path.join(args.text_folder,args.text_filename)
    df = pd.read_csv(fp)
    print('Reading inference file : {}'.format(fp))
    print('total docs for inference : {}'.format(len(df)))
    #%%
    M = TM(args.TOPIC_MODEL_OUT_PATH)
    #%%
    new_df = M.batch_predict_topic(df,doc_col_name='par',
                                   batch_size=1000,
                                   cache_dir=args.cache_folder,
                                   batch_id_range=args.batch_id_range)
    #%%

    #%%

    # #%%
    # t,p = M.predict_topic(df['par'][:500])
    # #%%
    # p_max_id = np.argmax(p,axis=1)
    # max_prob = p[np.arange(len(p_max_id)), p_max_id]
    # prob_df = pd.DataFrame(p)
    # t_data = {'org_topic_id': t, 'prob_topic_id': p_max_id,'max_probability':max_prob}
    # topic_df = pd.DataFrame(t_data)
    # # Concatenate DataFrames in sequence
    # final_df = pd.concat([df[:500], topic_df, prob_df], axis=1,ignore_index=True)
    # #%%
    # final_df.columns =list(df.columns)+list(topic_df.columns)+list(prob_df.columns)
    # #%%

# %%
