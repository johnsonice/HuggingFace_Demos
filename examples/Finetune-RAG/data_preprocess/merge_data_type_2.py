# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 20:07:38 2023

@author: CHuang
"""
### merge data type 1
import pandas as pd 
import os, sys



#%%
if __name__ == "__main__":

    folder_path = 'C:/Users/chuang/International Monetary Fund (PRD)/Liu, Yang - data/Data_Cleaning_CH/raw_type_2'
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
    combined_df['unique_paragraph'] = combined_df.index.astype(str) + "_" + combined_df['paragraph']
    
    #%%
    df_long = pd.wide_to_long(combined_df, 
                          stubnames=['question', 'answer'], 
                          i=['unique_paragraph','paragraph','report name'], 
                          j='sequence', 
                          sep=' ', 
                          suffix='\d+')
    df_long = df_long.reset_index()
    ## export raw to excile 
    df_long.to_excel('../data_processed/type2_combined.xlsx', index=False)
    
    #%%
    ### do some filtering and export filtered data 
    df_cleaned = df_long.dropna(subset=['question'])
    df_cleaned = df_cleaned.drop_duplicates(subset=['paragraph','question'])
    df_cleaned.to_excel('../data_processed/type2_combined_filtered.xlsx', index=False)
    