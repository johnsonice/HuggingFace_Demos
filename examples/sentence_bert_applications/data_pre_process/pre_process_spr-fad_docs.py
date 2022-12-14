#%%
import os,re,sys
sys.path.insert(0,'../../../libs')
sys.path.insert(0,'../')
from utils import get_all_files
import config
from tqdm import tqdm

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

#%%
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

#%%
if __name__ == "__main__":
    
    ## overall global variables 
    data_folder = os.path.join(config.data_folder,'Data','Raw_LM_Data','RAW')
    input_folder = os.path.join(data_folder,'BERT_PRE_TRAIN_DATA','SPR-FAD_All')
    out_folder = os.path.join(data_folder,'BERT_PRE_TRAIN_DATA_CLEAN','SPR_FAD')

    keep_tags = ['<Para>','<Title>','<Footnote>']

    input_files = get_all_files(input_folder,'.txt')
    for input_f in tqdm(input_files):
        fn = os.path.basename(input_f)
        fn_out = os.path.join(out_folder,fn)
        content = process_txt(input_f,keep_tags)
        export_to_txt(fn_out,content)


#%%