#%%
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
from transformers import pipeline
import os 
from datasets import load_dataset,load_from_disk
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import pyarrow.csv as pac
#%%
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def add_columns(example):
    example["text_processed"] = preprocess(example['text'])
    return example

def transform_pipe_results(res):   
    scores = [d['score'] for d in res]
    label = scores.index(max(scores))
    scores.append(label)
    
    return scores
#%%
from datasets import Features, Value
ft = Features({'created_at':Value('string'),
    'lang':Value('string'),
    'id':Value('string'),
    'text':Value('string')})


#%%
if __name__ == "__main__":    
    ### use lattest roberta based twitter sentiment model 
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    Inference_data_folder = '/home/chuang/Dev/DATA/tweet_by_year'
    Inference_data_out_dir = '/home/chuang/Dev/DATA/tweet_by_year_res/{}'
    Raw_Columns = ['created_at','lang','id','text']
    #%%
    config = AutoConfig.from_pretrained(MODEL)
    print(config.id2label)
    pipe = pipeline(task="text-classification",
                    model = MODEL,
                    tokenizer=MODEL,return_all_scores=True)#,device=0)
    #%%
    all_in_files = os.listdir(Inference_data_folder)
    for f_name in all_in_files:
        Inference_data_dir = os.path.join(Inference_data_folder,f_name)
        print('\n\n......processing {} ......\n\n'.format(f_name))
        chunk_dataset = load_dataset('csv', 
                                    data_files=Inference_data_dir,
                                    features=ft)
        print(chunk_dataset['train'][10])
        chunk_dataset = chunk_dataset.map(add_columns)
        chunk_dataset = chunk_dataset.filter(lambda example:len(example['text_processed'])>5)
        chunk_dataset = chunk_dataset['train']
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
        chunk_dataset.to_csv(Inference_data_out_dir.format(f_name))

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