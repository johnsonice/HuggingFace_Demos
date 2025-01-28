"""
Post process topic modeling infernce results 
- recategorize non-program review docs to non-program topics 
- manually identify certain categories"
    - digitial money 434
    - gender 435

"""
#%%
import os,sys
sys.path.insert(0,'../../libs')
import pandas as pd
from utils import accumulate_csv_files
import re
from search_util import get_keywords_groups,construct_rex,find_exact_keywords
import numpy as np
#%%

def process_keywords_with_logic(seasrch_key_files,return_merged_keys=True,
                                return_logic=True,and_key='\+',key_lower=True,
                                search_sheet_name=None):
    """
    Parameters
    ----------
    seasrch_key_files : file path; string
    Returns
    -------
    kg : Dict
        a dict of keywords groups 
    all_keys : List
        all search key merged together
    """
    kg = get_keywords_groups(seasrch_key_files,lower=key_lower,sheet_name=search_sheet_name)
    keys_nested = list(kg.values())
    
    ## remove duplicates 
    all_keys = list(set([item for sublist in keys_nested for item in sublist]))
    
    ## break keys with logics 
    filtered_list = [item for item in all_keys if and_key not in item]
    logical_keys = [item for item in all_keys if item not in filtered_list]
    
    ## process logical keys and merge them back 
    if len(logical_keys)>0:
        logical_keys_split = [item.split(and_key) for item in logical_keys]
        logical_keys_split = list(set([item.strip() for sublist in logical_keys_split for item in sublist]))
        
        filtered_list.extend(logical_keys_split) ## merge logical terms back togeher 
        filtered_list = list(set(filtered_list)) ## remove duplicates again 
        
    return kg,filtered_list, logical_keys

def capitalize_special(input_string):
    ## change string to be camle case
    if isinstance(input_string, str):
        input_string = input_string.lower()
        # Function to capitalize the matched character
        def capitalize_match(match):
            return match.group(0).upper()

        # Regular expression to match word boundaries followed by an alphanumeric character
        pattern = r'(?<=\s|\?|-|/)[a-zA-Z]|^[a-zA-Z]'
        
        # Use re.sub() to replace matches with their capitalized form
        capitalized_string = re.sub(pattern, capitalize_match, input_string)
    else:
        capitalized_string = input_string
    return capitalized_string


if __name__ =="__main__":
    ## define file pathes ##
    data_folder = '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000'
    input_folder = os.path.join(data_folder,'results_cache')
    topic_map_path = os.path.join(data_folder,'topic_v2_merged_info_customized.csv')
    topic_map_path_v2 = os.path.join(data_folder,'topic_v2_merged_info_customized_clean.csv')

    country_meta_path = os.path.join(data_folder,'other_data','country_map.xlsx')
    document_meta_path = os.path.join(data_folder,'other_data','All_AIV_2008-2023_meta.xlsx')
    keywords_path = os.path.join(data_folder,'keywords','search_terms.xlsx')

    merged_out_path = os.path.join(data_folder,'topic_v2_merged_agg_data.csv')
    merged_out_simple_path = os.path.join(data_folder,'topic_v2_merged_agg_data_simple.csv')

    ## read input files ##
    topic_map = pd.read_csv(topic_map_path)
    country_map = pd.read_excel(country_meta_path)
    country_map=country_map[['country', 'ifscode', 'income', 'ISO-3 code', 'ISO-2 code','REO Region']]
    doc_map = pd.read_excel(document_meta_path)

    #%%
    ### identify jointed articles 
    # Check if the text column contains "review", "arrangement", or "request"
    doc_map['joint_flag'] = doc_map['Title'].str.contains(r'\b(review|arrangement|request)\b', case=False, na=False)
    doc_map[doc_map['joint_flag'] ==1].head()

    ## rename topic meta files, fixing typos 
    topic_map.columns = ['Topic','Count','Name','CustomName','level_0','level_1','level_2']
    columns_to_transform = ['CustomName','level_0','level_1','level_2']
    for c in columns_to_transform:
        topic_map[c] = topic_map[c].apply(capitalize_special)
    #topic_map.head()
    topic_map.to_csv(topic_map_path_v2,index=False)


    ## get all program related topic ids
    program_topic_ids = topic_map[topic_map['level_0']=='Program']['Topic'].tolist()
    program_topic_ids = [str(t) for t in program_topic_ids]
    #%%

    ###### read raw data #######
    df = accumulate_csv_files(input_folder)
    ## get all non-program prob as a seperate df
    prob_df = df.loc[:,"0":"433"]
    prob_df = prob_df.drop(columns=program_topic_ids)
    ## get new non-program topic id by take the max prob within non-program topics 
    max_prob_topic_ids = prob_df.idxmax(axis=1).astype(int) ## it will give the max column name  ## this is different from prob_df.values.argmax(axis=1)
    max_prob_values = prob_df.max(axis=1)
    ## creatr new non-program topic ids
    df['non-program_topic_id'] = max_prob_topic_ids
    df['non-program_max_probability'] = max_prob_values

    #%%
    ## merge document and country metadata info to raw data 
    merged_df = df.merge(doc_map, left_on='File_Name', right_on='File Name', 
                         how='left',indicator=False)
    merged_df.drop(columns=['File Name'], inplace=True)
    merged_df = merged_df.merge(country_map, left_on='Country Code', 
                                right_on='ifscode', how='left',indicator=True)
    ############
    ## double check again on country code meta file, fix issues 
    ############
    #%%
    ## check on missing
    missing_in_df2 = merged_df[merged_df['_merge'] == 'left_only']
    print("Missing in df2:")
    #print(missing_in_df2)
    print(missing_in_df2['Country Code'].unique())  ## need to fix them latter 
    ### array([352, 579, 126, 163, 312,  -1, 309, 533, 351])
    #%%
    ###########################
    ## update revised topic id, when non-program reports, reclassify them
    merged_df['revised_topic_id'] = merged_df['prob_topic_id']
    merged_df.loc[merged_df['joint_flag']==False,'revised_topic_id'] = merged_df['non-program_topic_id']

    ### update topic id with keywords search 
    and_key = '+'  ## define and logic keys in provided keywords data 
    keywords_dict , all_search_keywords, logical_keys = process_keywords_with_logic(keywords_path,and_key=and_key)
    search_rex = construct_rex(all_search_keywords,casing=False,plural=False)  ## here we are using case insensitive
    merged_df['search_res'] = merged_df['par'].str.lower().apply(find_exact_keywords,rex=search_rex,return_count=False) 
    keywords_df = pd.json_normalize(merged_df.pop('search_res')).fillna(0)
    matched_cols = [k for k in keywords_df.columns if k in all_search_keywords] ## all are lower case, so we are fine here  
    keywords_df[matched_cols] = keywords_df[matched_cols].applymap(lambda x: 1 if x >= 1 else 0) ## turn all match to dummies 
    #%%
    ## apply logicals based on keywords patterns 
    for lk in logical_keys:
        ks = [k.strip() for k in lk.split(and_key)]
        if all([k in matched_cols for k in ks]):
            keywords_df[lk] = keywords_df[ks].sum(axis=1)
            keywords_df[lk] = np.where(keywords_df[lk] == len(ks), 1, 0)
        else:
            keywords_df[lk] = 0 
    ## drop original columns 
    #for k in ks:
    #    if k in matched_cols:
    #        df.drop([k],axis=1)

    #### change it to be more modular when having multiple group of keywords 
    digital_columns = [k for k in keywords_dict['digital_money'] if k in keywords_df.columns]
    gender_columns = [k for k in keywords_dict['gender'] if k in keywords_df.columns]
    merged_df['digital_money'] = keywords_df[digital_columns].sum(axis=1).clip(upper=1).astype(int)
    merged_df['gender'] = keywords_df[gender_columns].sum(axis=1).clip(upper=1).astype(int)
    #%%
    ### update and add new topic ids for gender and digital_money 
    # digital_money= 434, gender = 435
    merged_df['revised_topic_id_v2'] = merged_df['revised_topic_id'] 
    merged_df.loc[merged_df['digital_money'] == 1, 'revised_topic_id_v2'] = 434
    merged_df.loc[merged_df['gender'] == 1, 'revised_topic_id_v2'] = 435

    ## merge metadata info 
    merged_df['final_topic_id'] = merged_df['revised_topic_id_v2'] 
    #### merge topic information over 
    merged_df = merged_df.merge(topic_map, left_on='final_topic_id', right_on='Topic', how='left',indicator=False)

    #%%
    ## export to file 
    keep_columns = ['index','File_Name','Title', 'Country Code', 'Country_Name','Year','income','REO Region','par',
            'org_topic_id', 'prob_topic_id', 'max_probability','non-program_topic_id', 'non-program_max_probability','revised_topic_id','final_topic_id',
            'CustomName', 'level_0', 'level_1', 'level_2']
    merged_df.to_csv(merged_out_path,index=False)
    simple_merged_df = merged_df[keep_columns]
    simple_merged_df.to_csv(merged_out_simple_path,index=False)

    print('finished export data to {}'.format(merged_out_simple_path))
# %%
