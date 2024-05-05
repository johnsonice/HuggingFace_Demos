# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 20:30:39 2023

@author: CHuang
"""

## data merge for qa retrieval

import pandas as pd 
import os, sys

#%%
if __name__ == "__main__":
    
    root_f = r'C:\Users\chuang\International Monetary Fund (PRD)\Liu, Yang - data\Data_Cleaning_CH\data_processed'
    ## work on file type 1 
    f1 = os.path.join(root_f,'type1_combined_filtered.xlsx')
    
    keep_cols = ['query','text','answer']
    cols_name = ['question','context','answer']
    df = pd.read_excel(f1)
    df = df[keep_cols]
    df.columns = cols_name
    
    #%%
    ## work on file type 2
    f2 = os.path.join(root_f,'type2_combined_filtered.xlsx')
    df2 = pd.read_excel(f2)
    keep_cols = ['question','paragraph','answer']
    cols_name = ['question','context','answer']
    df2 = df2[keep_cols]
    df2.columns = cols_name
    
    #%%
    ## work on file type 3
    f3 = os.path.join(root_f,'type3_cleaned.xlsx')
    df3 = pd.read_excel(f3)
    
    #%%
    ## merge everything and clean
    df_all = pd.concat([df,df2,df3],ignore_index=True)
    df_all = df_all[cols_name]
    df_all = df_all.drop_duplicates(subset=['question','context']) 
    df_all.to_excel(os.path.join(root_f,'retrieval_train_data_clean.xlsx'), index=False)