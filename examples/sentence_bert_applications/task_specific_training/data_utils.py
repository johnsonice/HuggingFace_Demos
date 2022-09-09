
#%%
import os, csv, gzip, random
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample

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

#%%

if __name__ == "__main__":
    
    data_path = data_path = '/home/chuang/Dev/DATA/raw_data'
    nli_dataset_path = os.path.join(data_path,'AllNLI.tsv.gz')
    ## download data 
    get_sbert_nli_data(nli_dataset_path)

    ## read data 
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        train_dataset = nli2triplet_sets(reader,filter_key='split',split='train')

    print('Print one dict item')
    print(next(iter(train_dataset.items())))

    ## transform to triplet input examples 
    train_inputs_samples = transform2triplet_inputexamples(train_dataset)

# %%
