# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:16:57 2022

@author: CHuang
"""

import os,re,sys
from tqdm import tqdm
import pathlib
import joblib
from joblib import Parallel, delayed

def get_all_files(dirName,end_with=None): # end_with=".json"
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(fullPath)
        else:
            allFiles.append(fullPath)
    
    if end_with:
        end_with = end_with.lower()
        allFiles = [f for f in allFiles if pathlib.Path(f).suffix.lower() == end_with ] 

    return allFiles  

def read_txt(fp):
    ### read txt into lines ###
    with open(fp,'r',encoding='UTF-8',errors='ignore') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n').strip() for line in lines]
    #lines = [line for line in lines if len(line)>0]
    return lines

def export_to_txt(file_name,text_list):
    with open(file_name, 'w',encoding='utf8') as f:
        for t in text_list:
            try:
                f.write('%s\n' % t)
            except Exception as e:
                print(t)
                raise Exception(e)
    return None

def _clean_paragraph(para):
    #replace_num = re.compile(r'\(.*?\)')
    para = re.sub(r'<(.*?)>','',para.strip(),1) ## replace first instance of <>
    res = re.sub(r'^[\d]+.\s|\(.*?\)|\n','',para.strip()) ## clean leanding #.
    return res 

def process_txt(fp,keep_tags=None):
    lines = read_txt(fp)
    res_lines = []

    for l in lines:
        keep_flag = True

        ## length check 
        if len(l)>30:
            pass
        else:
            keep_flag = False
            continue
        
        ## tag track 
        if l[0] == '<':
            ls = l.split('>')
            if ls[0]+'>' in keep_tags:
                pass
            else:
                keep_flag = False
                continue
        
        if keep_flag:
            l = _clean_paragraph(l)
            res_lines.append(l)

    return res_lines


def multi_process_func(input_f,out_folder,keep_tags):
    
    """
    delayed process function for joblib 
    """
    try:
        fn = os.path.basename(input_f)
        fn_out = os.path.join(out_folder,fn)
        content = process_txt(input_f,keep_tags)
        export_to_txt(fn_out,content)
        return True
    except:
        return input_f

#%%
if __name__ == "__main__":
    
    MULTI = True
    ## overall global variables 
    #data_folder = os.path.join(config.data_folder,'Data','Raw_LM_Data','RAW')
    data_folder = os.path.join('/data/chuang/Language_Model_Training_Data/Data/Raw_LM_Data')
    input_folder = os.path.join(data_folder,'RAW_Large')
    out_folder = os.path.join(data_folder,'CLEAN_Large')
    keep_tags = ['<Para>','<Title>','<Footnote>']
    input_files = get_all_files(input_folder,'.txt')
    #%%
    #input_files = input_files[:1000]
    if MULTI:
        number_of_cpu = joblib.cpu_count() - 2 
        parallel_pool = Parallel(n_jobs=number_of_cpu,verbose=5)
        all_args = zip(input_files,[out_folder]*len(input_files),[keep_tags]*len(input_files))
        delayed_funcs = [delayed(multi_process_func)(inf,outfo,ktgs) for inf,outfo,ktgs in all_args]
        f_result = parallel_pool(delayed_funcs)
    else:
        for input_f in tqdm(input_files):
            fn = os.path.basename(input_f)
            fn_out = os.path.join(out_folder,fn)
            content = process_txt(input_f,keep_tags)
            export_to_txt(fn_out,content)