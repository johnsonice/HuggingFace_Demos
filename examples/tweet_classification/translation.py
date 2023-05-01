#%%
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
from transformers import pipeline
import os,argparse
from datasets import load_dataset,load_from_disk
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import pyarrow.csv as pac

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset

#### follow https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

#%%
class TranslationPipeline:
    def __init__(self, pretrained_model, tokenizer):
        self.model = pretrained_model
        self.tokenizer = tokenizer

    def __call__(self, sentences, src_lang, tgt_lang,batch_size=None):
        if batch_size:
            res_list = []
            input_batches = self.batch_inputs(sentences,batch_size)
            for batched_sentences in tqdm(input_batches):
                translated_sentences = self.translate(batched_sentences, src_lang, tgt_lang) 
                res_list.extend(translated_sentences)
            return res_list
        
        return self.translate(sentences, src_lang, tgt_lang)
    
    @staticmethod
    def batch_inputs(input_list, batch_size=16):
        # Calculate the total number of batches
        num_batches = (len(input_list) + batch_size - 1) // batch_size
        # Split the input_list into smaller batches
        return [input_list[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    

    def translate(self, batched_sentences, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        encoded_inputs = self.tokenizer(batched_sentences, return_tensors="pt",
                                        padding='longest',truncation=True,
                                        max_length=256)
        generated_tokens = model.generate(
            **encoded_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
        translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return translated_texts

#%%
#%%
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t =  '' if t.startswith('@') and len(t) > 1 else t # instad of '@user', replace to ''
        t = '' if t.startswith('http') else t # instead of 'http' replace to ''
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

def maybe_create_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' already exists.")
#%%
from datasets import Features, Value
ft = Features({'created_at':Value('string'),
    'lang':Value('string'),
    'id':Value('string'),
    'text':Value('string')})

def t_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', '--in_file', action='store', dest='in_file',
                    default='/data/chuang/Twitter_Data/Twitter_API_PY/Data/Tweet_by_user/chunks/User_Comments/France__comments.csv',
                    type=str) #v1_results\ArgentinaBrazilUruguayParaguay #user_tweets_filtered
    parser.add_argument('-out_file', '--out_file', action='store', dest='out_file',
                    default='/data/chuang/Twitter_Data/Twitter_API_PY/Data/Tweet_by_user/chunks/User_Comments_english/France__comments_eng.csv',
                    type=str)
    parser.add_argument('-src', '--source_language', action='store', dest='source_language',
                    default='fr_XX',type=str)  
    parser.add_argument('-tagt', '--target_language', action='store', dest='target_language',
                    default='en_XX',type=str)
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()    
        
    return args


##
## 

#%%
if __name__ == "__main__":    
    args = t_args()
    chche_dir = '/data/chuang/temp'
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                        cache_dir=chche_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                    cache_dir=chche_dir)
    T = TranslationPipeline(model,tokenizer)
    
    ## check file and folders 
    infer_file_dir = args.in_file
    assert os.path.exists(infer_file_dir)
    out_folder = os.path.dirname(args.out_file)
    maybe_create_folder(out_folder)

    ## load into dataset
    if os.path.splitext(infer_file_dir)[1] == '.xlsx':
        in_df = pd.read_excel(infer_file_dir)
        chunk_dataset = Dataset.from_pandas(in_df)
    elif os.path.splitext(infer_file_dir)[1] == '.csv':
        in_df = pd.read_csv(infer_file_dir)
        chunk_dataset = Dataset.from_pandas(in_df)

    print(chunk_dataset[:3])
    ## preprocess chunk dataset 
    chunk_dataset = chunk_dataset.map(add_columns)
    chunk_dataset = chunk_dataset.filter(lambda example:len(example['text_processed'])>6)
    ## translate from source language to target language 
    src_lang = args.source_language
    tgt_lang = args.target_language
    res = T(chunk_dataset['text_processed'], src_lang, tgt_lang,batch_size=32)
    res_df = pd.DataFrame(res)
    chunk_dataset=chunk_dataset.add_column('eng_text',res_df[0])
    chunk_dataset.to_csv(args.out_file)

#%%
    # #%%
    # all_in_files = os.listdir(Inference_data_folder)
    # all_in_files = [f for f in all_in_files if '.csv' in f]
    # #%%
    # for f_name in all_in_files:
    #     Inference_data_dir = os.path.join(Inference_data_folder,f_name)
    #     print('\n\n......processing {} ......\n\n'.format(f_name))
    #     chunk_dataset = load_dataset('csv', 
    #                                 data_files=Inference_data_dir,
    #                                 features=ft)
    #     print(chunk_dataset['train'][10])
    #     chunk_dataset = chunk_dataset.map(add_columns)
    #     chunk_dataset = chunk_dataset.filter(lambda example:len(example['text_processed'])>5)
    #     chunk_dataset = chunk_dataset['train']
    #     res = []
    #     for out in tqdm(pipe(KeyDataset(chunk_dataset, "text_processed"), 
    #                         batch_size=32, padding=True, 
    #                         truncation=True,max_length=256)):
    #         res.append(transform_pipe_results(out))
    #     res_df = pd.DataFrame(res)
    #     #%%
    #     ## {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    #     chunk_dataset=chunk_dataset.add_column('negative',res_df[0])
    #     chunk_dataset=chunk_dataset.add_column('netural',res_df[1])
    #     chunk_dataset=chunk_dataset.add_column('positive',res_df[2])
    #     chunk_dataset=chunk_dataset.add_column('label',res_df[3])
    #     ## remove original text to save space
    #     chunk_dataset = chunk_dataset.remove_columns(['text'])
    #     chunk_dataset.to_csv(Inference_data_out_dir.format(f_name))

