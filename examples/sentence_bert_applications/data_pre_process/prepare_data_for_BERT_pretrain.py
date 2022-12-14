# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:13:08 2019

@author: chuang
"""
import pickle
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from mp_utils import Mp

#%%
def _flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
            
def _clean_paragraph(para):
    #replace_num = re.compile(r'\(.*?\)')
    res = re.sub(r'^[\d]+.\s|\(.*?\)|\n','',para.strip())
    return res 

def transform_to_BERT(para):
    tl = sent_tokenize(para)
    if len(tl)>1 and all([len(l)>1 for l in tl]):
        res = "\n".join(tl)
        return res
    else:
        return None

def get_all_paragraphs(pl_folder,TEST=False):
    files =  os.listdir(pl_folder)    

    ## read all into one list 
    if TEST:
        files = files[:2]
    
    all_docs = []
    for f in files:
        print(f)
        data_path = os.path.join(pl_folder,f)
        docs = pickle.load(open(data_path, "rb"))
        all_docs.extend(docs)
    
    all_paras = list(_flatten(all_docs))
    
    return all_paras

def export_to_txt(file_name,text_list):
    with open(file_name, 'w',encoding='utf8') as f:
        for t in text_list:
            try:
                f.write('%s\n' % t)
            except Exception as e:
                print(t)
                raise Exception(e)
    
    return None

def export_to_txt_BERT(file_name,text_list):
    with open(file_name, 'w',encoding='utf8') as f:
        for t in text_list:
            try:
                f.write('%s\n\n' % t)
            except Exception as e:
                print(t)
                raise Exception(e)
    
    return None
#%%

if __name__ == "__main__":
    
    ## overall global variables 

    pl_folder = 'D:/usr-profiles/chuang/Desktop/Dev/textmining/2_imf_docs/1_use_xmls/pickle'

    all_paras = get_all_paragraphs(pl_folder)
    all_paras = [_clean_paragraph(p) for p in all_paras]
    #%%
    export_to_txt('raw.txt',all_paras)
    #%%
    transformed_paras = [transform_to_BERT(p) for p in all_paras]
    transformed_paras = [p for p in transformed_paras if p is not None]
    export_to_txt_BERT('train_data.txt',transformed_paras)

    