#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:17:59 2022

@author: chengyu
"""
import os
import pandas as pd
import config
from tqdm.auto import tqdm


climate_keys = ['policy', 'government', 'adaptation', 'mitigation',
    'climate adaptation', 'climate mitigation', 'non-market base', 'tax',
    'tax on fuel consumption', 'tariff', 'subsidies', 'consumption subsidy',
    'producer subsidy', 'fuel subsidy', 'preserve incentive',
    'electric vehicle incentive', 'tradable permit', 'liability rule',
    'deposit-refund', 'ban', 'performance standard',
    'environmental reporting', 'public adoption', 'investment', 'r & d',
    'energy', 'fiscal instrument', 'training', 'voluntary agreement',
    'risk', 'physical', 'physical risk', 'transition', 'transition risk',
    'financial', 'financial risk', 'liability', 'liability risk',
    'climate tag', 'policy tag', 'risk tag', 'climate policy',
    'climate policy risk', 'climate risk', 'monetary policy',
    'climate change', 'global warm', 'carbon', 'emissions',
    'carbon emission', 'greenhouse gas', 'sea-level rise', 'low carbon']

keep_cols = ['id','pub_date', 'source', 'language', 'title', 'body', 'Year', 'ym',
             'climate_count','climate_binary','netural', 'positive', 'negative', 'label']


def agg_by(df,group_cols:list,value_cols_dict:dict):
    """
    aggregate by group and agg values cols 

    """
    agg_df = df.groupby(group_cols).agg(value_dict)
    agg_df.columns = agg_df.columns.map("_".join)
    agg_df.reset_index(inplace=True)
    
    return agg_df


#%%
if __name__ == "__main__":    

    #MODEL_OUTDIR = os.path.join(config.model_folder,'news_classification')
    res_dir = os.path.join(config.data_folder,'Data','climate_news','inference_data','infer_results')
    out_agg_dir = os.path.join(config.data_folder,'Data','climate_news','inference_data','final_agg.csv')
    res_files = os.listdir(res_dir)
    res_files = [f for f in res_files if '~' not in f]

    ## agg columns 
    agg_res_dfs = []
    group_cols = ['ym','climate_binary','label']
    value_dict = {'label':['count']}
    
    for f in tqdm(res_files):
        df = pd.read_csv(os.path.join(res_dir,f), encoding='utf8')
        df['climate_count'] = df[climate_keys].sum(axis=1)
        df['climate_binary'] = df['climate_count']>0
        df = df[keep_cols]
        agg_df = agg_by(df,group_cols,value_dict)
        agg_res_dfs.append(agg_df)
    
    #%%
    final_agg_df = pd.concat(agg_res_dfs)
    final_agg_df = final_agg_df.groupby(group_cols)['label_count'].sum()
    label_map = {0:'neutral',1:'positive',2:'negative'}
    final_agg_df.name='label_count'
    final_agg_df = pd.DataFrame(final_agg_df)
    final_agg_df.reset_index(inplace=True)
    final_agg_df['label'].replace(label_map,inplace=True)
    final_agg_df.to_csv(out_agg_dir,encoding='utf8',index=False)
    print('save results to {}'.format(out_agg_dir))