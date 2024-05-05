# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:43:03 2023

@author: CHuang
"""
#%%

import os,sys,openai
import pandas as pd 
from data_utils import get_completion
from utils import load_json,exception_handler
from prompts import gen_summary_basic,gen_query_basic
import tqdm,re
import ast
import time
import os, sys , ssl

#%%
key = load_json('../../Keys/openai_key.json') 
os.environ['OPENAI_API_KEY'] = key['ChatGPT1']['API_KEY']
openai.api_key  = os.getenv('OPENAI_API_KEY')

#%%
@exception_handler(error_msg=None,error_return=None)
def gen_summary_on_context(prompt_template,context_info):
    response = get_completion(prompt=prompt_template['Human'].format(context_str=context_info),
                        sys_msg=prompt_template['System'],
                        model='gpt-3.5-turbo-16k')
    
    if prompt_template['parsing_func']:
        res = prompt_template['parsing_func'](response)
    else:
        raise Exception('Please define your result parsing function in prompt template')
    
    time.sleep(1) ## wait for a sec to alleviate server side errors 

    return res

@exception_handler(error_msg=None,error_return=None,attempts=3,delay=2)
def gen_query_on_question(prompt_template,user_question):
    response = get_completion(prompt=prompt_template['Human'].format(user_question=user_question),
                        sys_msg=prompt_template['System'],
                        model='gpt-3.5-turbo-16k')
    
    if prompt_template['parsing_func']:
        res = prompt_template['parsing_func'](response)
    else:
        raise Exception('Please define your result parsing function in prompt template')
    
    time.sleep(1) ## wait for a sec to alleviate server side errors 
    
    return res

#%%
if __name__ == "__main__":
    
    ## load raw context data 
    data_folder = r'../to_server_data'
    human_data_path = os.path.join(data_folder,'retrieval_train_data_clean_manual_inspect.xlsx')
    gpt_data_path = os.path.join(data_folder,'retrieval_train_data_gpt_manual_inspect.xlsx')
    human_df = pd.read_excel(human_data_path,sheet_name='Long')
    human_df_short = pd.read_excel(human_data_path,sheet_name='Short')
    #%%
    # example_text = human_df['context'].iloc[0]
    # res = gen_summary_on_context(gen_summary_basic,example_text)
    # print(res)
    # %%
    ### generate queries based on natural language questions 
    all_questions = list(set(human_df['question'].tolist() + human_df_short['question'].tolist()))

    #%%

    # response = gen_query_on_question(gen_query_basic,all_questions[110])
    # print(response)

    #######

    q2q_map = {}
    for q in tqdm.tqdm(all_questions):
        gq_dict = gen_query_on_question(gen_query_basic,q)
        q2q_map[q]=gq_dict

    #######













    ######################################

    #%%
    ### summarize all long text --------------------------
    res_list = []
    for context_info in tqdm.tqdm(human_df['context']):
        summary = gen_summary_on_context(gen_summary_basic,context_info)
        res_list.append(summary)

    #%% process merge 
    print('org length : {}'.format(len(human_df)))
    human_df.rename(columns={'context':'context_org'},inplace=True)
    human_df['context'] = res_list
    human_df=human_df.dropna(subset=['context'])
    print('after length : {}'.format(len(human_df)))
    #%%
    human_df_short['context_org']=human_df_short['context']
    result_df = human_df.append(human_df_short, ignore_index=True).sort_values(by='question')
    print('total length : {}'.format(len(result_df)))

    outpath = os.path.join(data_folder,'V2','retrieval_train_data_clean.xlsx')
    result_df.to_excel(outpath)
    print('output to {}'.format(outpath))

# %%
