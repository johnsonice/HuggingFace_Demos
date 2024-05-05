# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:07:57 2023

@author: CHuang
"""

import pandas as pd 
import os, sys

#%%
if __name__ == "__main__":
    
    data_folder = r'C:\Users\chuang\International Monetary Fund (PRD)\Liu, Yang - data\Data_Cleaning_CH\data_GPT_Generated/batches'
    output_folder = r'C:\Users\chuang\International Monetary Fund (PRD)\Liu, Yang - data\Data_Cleaning_CH\data_GPT_Generated'
    ## work on file type 1 
    data_files = [os.path.join(data_folder,f) for f in os.listdir(data_folder) if f.endswith('.xlsx') or f.endswith('.xls')]
    
    keep_cols = ['question','context','answer']
    
    all_data = pd.DataFrame()
    for ef in data_files:
        df = pd.read_excel(ef)
        df = df[keep_cols]
        all_data = all_data.append(df, ignore_index=True)

    all_data_unique = all_data.drop_duplicates(subset=['question','context'])
    all_data_unique.to_excel(os.path.join(output_folder,'retrieval_train_data_gpt.xlsx'))
    
    
    