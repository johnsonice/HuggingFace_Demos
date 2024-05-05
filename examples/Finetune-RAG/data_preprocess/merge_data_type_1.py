# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:27:02 2023

@author: CHuang
"""

### merge data type 1
import pandas as pd 
import os, sys



#%%
if __name__ == "__main__":

    folder_path = 'C:/Users/chuang/International Monetary Fund (PRD)/Liu, Yang - data/Data_Cleaning_CH/raw_type_1'
    all_files = os.listdir(folder_path)
    
    dfs = []
    for file in all_files:
        if file.endswith('.xlsx') or file.endswith('.xls'):  # Check if the file is an Excel file
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)
            
    combined_df = pd.concat(dfs, ignore_index=True)
    
    #### process combined data 
    combined_df = combined_df.drop(columns=combined_df.filter(like='Unnamed').columns)
    combined_df['unique_query'] = combined_df.index.astype(str) + "_" + combined_df['query']
    
    #%%
    df_long = pd.wide_to_long(combined_df, 
                          stubnames=['docid', 'filename','text','relevancy_score','answer'], 
                          i=['unique_query','query','filter'], 
                          j='sequence', 
                          sep='_', 
                          suffix='\d+')
    df_long = df_long.reset_index()
    ## export raw to excile 
    df_long.to_excel('../type1_combined.xlsx', index=False)
    
    #%%
    ### do some filtering and export filtered data 
    df_cleaned = df_long.dropna(subset=['text'])
    df_cleaned = df_cleaned[df_cleaned['relevancy_score']>0]
    df_cleaned = df_cleaned.drop_duplicates(subset=['query','text'])
    df_cleaned.to_excel('../type1_combined_filtered.xlsx', index=False)
    