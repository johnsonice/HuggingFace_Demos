#%%
### topic merging based on manual labeled categories 

## inference script for AIV topic identification exercise 




import pandas as pd
from tqdm import tqdm
from topic_inference import TM
import os ,re,sys,argparse
import nltk 

def flatten(nested_list):
    """
    Flattens a nested list.
    """
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten(element))
        else:
            flat_list.append(element)
    return flat_list

def _extract_tag(s):
    if isinstance(s,str):
        pattern = r'<(.*?)>' # Regular expression pattern to match the content inside the first pair of angle brackets
        match = re.search(pattern, s) # Search for the pattern in the string
        if match:   # If a match is found, return the content inside the brackets
            return match.group(1)
        else:
            return None
    else:
        return None

def _filter_by_tag(s,keep_tags=['title','para']):
    tag = _extract_tag(s)
    if tag:
        if tag.lower() in keep_tags:
            return True
    return False

def clean_text(text):
    return re.sub(r'^<.*?>\s*', '', text)

def process_df(input_df):
    '''
    Do some simple filtering by keep only para tag and long paragraphs 
    '''
    input_df['tag_filter']=input_df['par'].apply(_filter_by_tag,args=(['para'],))  
    input_df['length_filter'] = input_df['par'].apply(lambda s: len(s.split())>15)
    input_df = input_df[(input_df['tag_filter']) & (input_df['length_filter'])]
    input_df = input_df.drop(columns=['tag_filter','length_filter'])
    input_df['par'] = input_df['par'].apply(clean_text)

    return input_df

def accumulate_csv_files(directory,end_with='.csv',process_func=None):
    """
    This function reads all CSV files from a given directory and appends them into one DataFrame.
    """
    # List to hold dataframes
    dataframes = []
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(end_with):
            filepath = os.path.join(directory, filename)
            # Read the CSV file and append to the list
            df = pd.read_csv(filepath)
            if process_func:
                df = process_func(df)
            dataframes.append(df)
    # Concatenate all dataframes in the list
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

def model_args(args_list=None):
    MODEL_OUTPUT = '/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000'
    SBERT_CHECKPOINT = '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000'
    INPUT_DATA_FOLDER ='/data/chuang/Dev/Authorities_View/Traction/data/AIV_CSV_RAW/All_AIV_2008-2023_csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--molder_folder', action='store', dest='MODEL_OUTPUT',
                    default=MODEL_OUTPUT,type=str)
    parser.add_argument('-tmo', '--topic_model_out_path', action='store', dest='TOPIC_MODEL_OUT_PATH',
                    default=os.path.join(MODEL_OUTPUT,'topic_model_v2_mreged'),type=str)
    parser.add_argument('-txf', '--text_folder', action='store', dest='text_folder',
                    default=INPUT_DATA_FOLDER,type=str)
    parser.add_argument('-cache', '--cache_folder', action='store', dest='cache_folder',
                    default=os.path.join(MODEL_OUTPUT,'results_cache'),type=str)
    parser.add_argument('-batch_id_range', '--batch_id_range', action='store', dest='batch_id_range',
                        default="0-10000",type=str)
    
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    

    return args


#%%

if __name__ == "__main__":
    #startTime = time.time()
    args = model_args()
    #%%
    df = accumulate_csv_files(args.text_folder,process_func=process_df)
    #%%
    print('Reading inference file : {}'.format(args.text_folder))
    print('total docs for inference : {}'.format(len(df)))
    #%%
    M = TM(args.TOPIC_MODEL_OUT_PATH)
    #%%
    new_df = M.batch_predict_topic(df,doc_col_name='par',
                                   batch_size=500,
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
