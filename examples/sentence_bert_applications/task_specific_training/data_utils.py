
#%%
import os, csv, gzip, random
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import datasets

def get_sbert_nli_data(nli_dataset_path):
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

def add_to_samples(train_data, sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)

def nli2triplet_sets(data_iter,filter_key=None,split=None):
    ## transform raw NLI data format, into {'contradiction': set(), 'entailment': set(), 'neutral': set()} style
    train_data={}
    
    ## convert dataset to iter 
    if isinstance(data_iter, datasets.arrow_dataset.Dataset):
        data_iter = iter(data_iter)
  
    for row in data_iter:
        if filter_key and split:
            if row[filter_key]!=split:
                continue
        sent1 = row['sentence1'].strip()
        sent2 = row['sentence2'].strip()
        add_to_samples(train_data,sent1, sent2, row['label'])
        add_to_samples(train_data,sent2, sent1, row['label'])  #Also add the opposite
    return train_data

def transform2triplet_inputexamples(nlidata_dict):
    ## format data into triplet setting , one anchor, one positive and one negative
    ## in this example, we will use multiple negatives ranking loss with hard negative, it takes a triplet 
    ## see : https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/nli

    train_samples = []
    for sent1, others in nlidata_dict.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))
    
    return train_samples

def load_tranform_HF_nli_datasets(ds_name,keep_cols=['premise','hypothesis','label'],split='train'):
    
    ## loda ds
    nli = datasets.load_dataset(ds_name, split=split)
    ## keep useful columns
    remove_cols = list(set(nli.column_names) - set(keep_cols))
    nli = nli.remove_columns(remove_cols)
    ## filter and transform lable to be consistant
    nli = nli.filter(lambda e: e['label'] in [0,1,2])
    label_map = {i:v for i,v in enumerate(nli.info.features['label'].names)} ##{0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    nli = nli.map(lambda example: {'new_label':label_map[example['label']]},remove_columns=['label']) #,batched=True
    ## rename columns 
    nli=nli.rename_column('new_label','label')
    nli=nli.rename_column('premise','sentence1')
    nli=nli.rename_column('hypothesis','sentence2')
    ## return 
    
    return nli

def load_all_nli_from_HF(nli_ds_names=['multi_nli','anli','snli'],split='train'):
    all_nli = []
    ## load all nli datasest
    for ds_n in nli_ds_names:
        if ds_n == 'anli':
            ## anli has differnet splits 
            for r in ['_r1','_r2','_r3']: 
                ds = load_tranform_HF_nli_datasets(ds_n,split=split+r)
        else:
            ## notmal train split 
            ds = load_tranform_HF_nli_datasets(ds_n,split=split) 
        
        all_nli.append(ds)
    ## concate all 
    all_ds = datasets.concatenate_datasets(all_nli)
    
    return all_ds

def get_HF_nli_dev_test(split_name='dev'):
    if split_name=='dev':
        all_nli = []
        all_nli.append(load_tranform_HF_nli_datasets('multi_nli',split='validation_matched'))
        for r in ['_r1','_r2','_r3']:
            all_nli.append(load_tranform_HF_nli_datasets('anli',split='dev' + r))
        all_nli.append(load_tranform_HF_nli_datasets('snli',split='validation'))
        all_ds = datasets.concatenate_datasets(all_nli)

    elif split_name=='test':
        all_nli_dev = []
        all_nli_dev.append(load_tranform_HF_nli_datasets('multi_nli',split='validation_mismatched'))
        for r in ['_r1','_r2','_r3']:
            all_nli_dev.append(load_tranform_HF_nli_datasets('anli',split='test' + r))
        all_nli_dev.append(load_tranform_HF_nli_datasets('snli',split='test'))
        all_ds = datasets.concatenate_datasets(all_nli_dev)
        
    hf_data = nli2triplet_sets(iter(all_ds))
    samples = transform2triplet_inputexamples(hf_data)
        
    return samples

#%%

if __name__ == "__main__":
    
    # data_path = data_path = '/home/chuang/Dev/DATA/raw_data'
    # nli_dataset_path = os.path.join(data_path,'AllNLI.tsv.gz')
    # ## download data 
    # get_sbert_nli_data(nli_dataset_path)

    # ## read data 
    # with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    #     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    #     train_dataset = nli2triplet_sets(reader,filter_key='split',split='train')

    # print('Print one dict item')
    # print(next(iter(train_dataset.items())))

    ## transform to triplet input examples 
    #train_inputs_samples = transform2triplet_inputexamples(train_dataset)
    
    ## Load from HF with 3 NLI datasets 
    ## anli; snli and multi_nli 
    # HF_datasets = load_all_nli_from_HF(nli_ds_names=['multi_nli','anli','snli'],split='train')
    # ## transform to the same train_data fromat 
    # train_data = nli2triplet_sets(iter(HF_datasets))
    # ## transform to same input examples 
    # train_samples = transform2triplet_inputexamples(train_data)
    
    ## Load test and dev data from 2 NLI datasets 
    dev_samples = get_HF_nli_dev_test(split_name='dev')
    test_samples = get_HF_nli_dev_test(split_name='test')

# %%
