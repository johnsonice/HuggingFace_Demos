#%%
#from sklearn.datasets import fetch_20newsgroups
import os, sys,ssl,argparse
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
from utils import get_all_files,txt2list
from tqdm import tqdm
import numpy as np
## in case you are behind a proxy 
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

## topic model packages
from bertopic import BERTopic
import gensim
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

## import arguments 
from topic_arguments import topic_model_args, train_args_space, topic_rep_args_space

#%%
def read_txt(f_p):
    txt_l = txt2list(f_p)
    txt_l = [t for t in txt_l if len(t.split())>5]
    return txt_l


def read_all_data(raw_txt_folder):
    out_list = []
    input_files = get_all_files(raw_txt_folder,'.txt')
    for inf in tqdm(input_files):
        txt_l = read_txt(inf)
        if len(txt_l)>1:
            out_list.extend(txt_l)
    return out_list


def model_setup(train_args):
    '''
    initialize topic model with training args 
    '''
    ## Step 3 - set up umap for reduction 
    ## if you want to make it reproducable; set random state in umap to be fixed 
    umap_model = UMAP(n_neighbors=train_args.n_neighbors,   # local neighborhood size for UMAP. default is 15, larget mean more global structure
                                                            # This is the parameter that controls the local versus global structure in data
                    n_components=train_args.n_components,   # output dimension for UMAP
                    min_dist=0,             # to allow UMAP to place points closer together (the default value is 1.0)
                    metric='cosine',        # use cosine distance 
                    random_state=42)        # fix random seed 

    ## Step 3 - Cluster reduced embeddings
    ## see link for more param selection:
    ## https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#parameter-selection-for-hdbscan
    hdbscan_model = HDBSCAN(min_cluster_size=train_args.min_cluster_size,  #the minimum number of documents in each cluster, for larger data, this should be larger
                            min_samples=train_args.min_samples,            #controls the number of outliers. It defaults to the same value as min_cluster_size. 
                                                            #The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, 
                                                            #and clusters will be restricted to progressively more dense areas
                                                            #we should keep this constant when tuning other parameters 
                            metric=train_args.metric,       #I guess we can try cosine here ? 
                            cluster_selection_method='eom', #The default method is 'eom' for Excess of Mass, the algorithm described in How HDBSCAN Works https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html.
                            prediction_data=True)

    ## Step 4-5 - prepare param for c-tfidf for topic representation 
    ## additional topic merging will be done by compare distance (based on bog method on c-tfidf), set to auto will use HDBSCAN
    ## remove stop words when fingding keys for topic representation ; sbert will still use full sentence 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2),
                                        stop_words="english",       # you can also provide a customized list 
                                        min_df=train_args.min_df,                  # set min number of word frequency
                                        #vacabulary=custom_vocab,   # you can also use a customized vocabulary list, 
                                                                    # e.g use keybert: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#keybert-bertopic
                                        )               
    ## ctfidf param can be pass in topicbert main function 


    ## call main function 
    topic_model = BERTopic(
                    umap_model=umap_model,              # Reduce dimensionality 
                    hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
                    vectorizer_model=vectorizer_model,  # Step 4,5 - use bang of words and ctfidf for topic representation
                    diversity= train_args.diversity,            # Step 6 - Diversify topic words ; maybe also try 0.5?
                    ## other params 
                    language="English",
                    verbose=True,
                    top_n_words=train_args.top_n_words,         # number of topic words to return; can be changed after model is trained 
                                                                # https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.update_topics
                    min_topic_size=train_args.min_cluster_size, # this should be the same as min_cluster_size in HDBSCAN
                    nr_topics='auto',               # number of topics you want to reduce to ; auto will use results from HDBSCAN on c-tfidf
                    calculate_probabilities = False, # Whether to calculate the probabilities of all topics per document instead of the probability of the assigned topic per document. 
                    )
    
    return topic_model 

def prepare_docs_for_coherence_eval(docs,topics,probabilities,model):
    documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics,
                          "Topic_prob": probabilities})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    #print(documents_per_topic.head())
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic) if words!=''] 
                for topic in range(len(set(topics))-1)]
    topic_words = [t for t in topic_words if len(t) >0] ## for some reason some topics has all "" as topic words

    return topic_words,tokens,corpus,dictionary

def get_coherence_score(topic_words,tokens,corpus,dictionary):
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v')
    coherence = coherence_model.get_coherence()
    
    return coherence

def eval_coherence_score(docs,topics,probabilities,model):
    topic_words,tokens,corpus,dictionary = prepare_docs_for_coherence_eval(docs,topics,probabilities,model)
    coherence = get_coherence_score(topic_words,tokens,corpus,dictionary)
    return coherence

def train_and_eval(args,docs,embeddings):
    topic_model = model_setup(args)
    topics, probabilities = topic_model.fit_transform(docs,embeddings)
    if isinstance(probabilities,np.ndarray):
        probabilities = probabilities.tolist()

    scores = eval_coherence_score(docs,topics,probabilities,topic_model)

    topic_freq = topic_model.get_topic_freq()
    outlier_percent = topic_freq['Count'][topic_freq['Topic'] == -1].iloc[0]/topic_freq['Count'].sum()
    
    ## you probably aldo don't want too many outliers 
    ## other than tune cluster size, you can also try reducer outliers after model is trained 
    #https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.reduce_outliers

    return scores,outlier_percent,topic_model

#%%
if __name__ == "__main__":
    args = topic_model_args()
    ## set paths 
    data_folder = args.data_folder
    input_folder = args.input_files_folder
    out_folder = args.out_folder
    emb_path = os.path.join(out_folder,'sentence_embeddings.npy')
    docs_path = os.path.join(out_folder,'docs.npy')
    result_path = os.path.join(out_folder,'results.csv')

    #input_files = get_all_files(input_folder,'.txt')
    LOAD_EMB = True
    TUNE = True

    ## set up topics models 
    if not LOAD_EMB:
        ## read raw documents 
        docs = read_all_data(input_folder)
        ## load model 
        sentence_model = SentenceTransformer("all-distilroberta-v1")                
        ## encode sentences 
        embeddings = sentence_model.encode(docs, show_progress_bar=True)
        assert len(docs)==len(embeddings)
        ## cache embedings 
        embeddings = np.array(embeddings)
        docs = np.array(docs)
        np.save(emb_path,embeddings)
        np.save(docs_path,docs)
    else:
        print('Load embeding from {}'.format(emb_path))
        embeddings = np.load(emb_path)
        docs = np.load(docs_path)
        assert len(docs)==len(embeddings)
        print('Number of docs: {}'.format(len(docs)))

    ## for testing purpose, try a small sample size 
    embeddings = embeddings[:50000]
    docs = docs[:50000]

    results = []
    if TUNE:
        for params in tqdm(train_args_space):
            args.__dict__.update(params)
            print(args)
            try:
                scores,outlier_percent,topic_model = train_and_eval(args,docs,embeddings)
                n_topics = len(topic_model.get_topic_freq())
                results.append(params.update(
                                                {
                                                    'score':scores,
                                                    'outlier_share':outlier_percent,
                                                    'n_topics': n_topics
                                                }
                                            )
                                )
                print('score: {} outlier share: {} number_topics: {}'.format(scores,outlier_percent,n_topics))
            except:
                print('-- Error -- {}'.format(params))
        res_df = pd.DataFrame(results)
        res_df.to_csv(result_path)
    else:
        print(args)
        scores,outlier_percent,topic_model = train_and_eval(args,docs,embeddings)
        n_topics = len(topic_model.get_topic_freq())
        print('score: {} outlier share: {} number_topics: {}'.format(scores,outlier_percent,n_topics))

 
