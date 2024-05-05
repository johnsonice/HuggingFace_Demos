#%%
## inference simple
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
from transformers import pipeline
import os, argparse
from datasets import load_dataset,load_from_disk
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
#%%
def preprocess(text):
    if not isinstance(text,str):
        return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def add_columns(example):
    example["eng_text_processed"] = preprocess(example['text'])
    return example

def transform_pipe_results(res):   
    scores = [d['score'] for d in res]
    label = scores.index(max(scores))
    scores.append(label)
    
    return scores

def t_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', '--in_file', action='store', dest='in_file',
                    default='/data/chuang/Twitter_Data/Twitter_API_PY/Data/Tweet_by_user/chunks/User_Comments_english/France__comments_eng.csv',
                    type=str)
    parser.add_argument('-out_file', '--out_file', action='store', dest='out_file',
                    default='/data/chuang/Twitter_Data/Twitter_API_PY/Data/Tweet_by_user/chunks/User_Comments_sentiment/France__comments_sentiment.csv',
                    type=str) #v1_results\ArgentinaBrazilUruguayParaguay #user_tweets_filtered
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    
        
    return args

from datasets import Features, Value
ft = Features({'created_at':Value('string'),
    'lang':Value('string'),
    'tweet_id':Value('string'),
    'country':Value('string'),
    'text':Value('string')})
#%%
if __name__ == "__main__":    
    args = t_args([])
    ### use lattest roberta based twitter sentiment model 
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    data_folder = '/data/chuang/Twitter_Data/Twitter_API_PY/Data/Tweet_by_user/chunks'
    Inference_data_dir = args.in_file
    Inference_data_out_dir = args.out_file
    #Raw_Columns = ['created_at','lang','tweet_id','country','text']

    #%%
    config = AutoConfig.from_pretrained(MODEL)
    print(config.id2label)
    pipe = pipeline(task="text-classification",
                    model = MODEL,
                    tokenizer=MODEL,return_all_scores=True)#,device=0)
    #%%
    print('\n\n......processing {} ......\n\n'.format(Inference_data_dir))
    # chunk_dataset = load_dataset('csv', 
    #                             data_files=Inference_data_dir,)
    # #                            features=ft)
    in_df = pd.read_csv(Inference_data_dir)
    chunk_dataset = Dataset.from_pandas(in_df)
    #%%
    print(chunk_dataset[10])
    #%%
    chunk_dataset = chunk_dataset.map(add_columns)
    chunk_dataset = chunk_dataset.filter(lambda example:len(example['eng_text_processed'])>5)
    #%%
    res = []
    for out in tqdm(pipe(KeyDataset(chunk_dataset, "text_processed"), 
                        batch_size=32, padding=True, 
                        truncation=True,max_length=256)):
        res.append(transform_pipe_results(out))
    res_df = pd.DataFrame(res)

    #%%
    ## {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    chunk_dataset=chunk_dataset.add_column('negative',res_df[0])
    chunk_dataset=chunk_dataset.add_column('netural',res_df[1])
    chunk_dataset=chunk_dataset.add_column('positive',res_df[2])
    chunk_dataset=chunk_dataset.add_column('label',res_df[3])
    ## remove original text to save space
    chunk_dataset = chunk_dataset.remove_columns(['text'])
    #%%
    chunk_dataset.to_csv(Inference_data_out_dir)
    # df = chunk_dataset.to_pandas()
    # df.to_csv(Inference_data_out_dir,encoding='utf8')
    #df.to_excel(os.path.join(data_folder,'climate_awareness_agg_senti.xlsx'))
    #%%
    #df.to_pickle(os.path.join(data_folder,'climate_awareness_agg_senti.pkl'))
    #df.to_pickle(os.path.join(data_folder,'climate_awareness_agg_senti.parquet'))
    #%%
    
    #%%
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # config = AutoConfig.from_pretrained(MODEL)
    # # PT
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # #%%
    # text = "Covid cases are increasing fast!"
    # text = preprocess(text)
    # #%%
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # scores = output[0][0].detach().numpy()
    # scores = softmax(scores)
    # # %%
    # ranking = np.argsort(scores)
    # ranking = ranking[::-1]
    # #%%
    # for i in range(scores.shape[0]):
    #     l = config.id2label[ranking[i]]
    #     s = scores[ranking[i]]
    #     print(f"{i+1}) {l} {np.round(float(s), 4)}")
    # # %%
    # ### use pip 
    # pipe = pipeline(task="text-classification",
    #                 model = MODEL,
    #                 tokenizer=MODEL,return_all_scores=True)#,device=0)
    # #%%
    # print(pipe(text))
    # print(transform_pipe_results(pipe(text)[0]))
    # #%%