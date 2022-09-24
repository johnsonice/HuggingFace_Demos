"""
Train SBERT model with customized loss function 

"""
#%%
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
import torch
from torch import nn

#%%

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self,model_name,tokenizer_name=None,normalize=True):
        super(AutoModelForSentenceEmbedding,self).__init__()
        if tokenizer_name is None:
            tokenizer_name=model_name

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedings(self,sentences:list,pooling_method='mean'):
        '''
        The whole process from tokenization to get pooled embeding
        '''

        kwargs = self.tokenizer(test_sentences,return_tensors="pt", truncation=True, padding=True) #padding="max_length", max_length=args.max_length,
        model_output = self.model(**kwargs)
        if pooling_method.lower() == 'mean':
            embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        elif pooling_method.lower() == 'cls':
            embeddings = self.cls_pooling(model_output)
        else:
            raise("{} pooling method not yet implemented".format(pooling_method))
        
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self,model_output,attention_mask):
        '''
        Average the last layer for input sentences as sentence embeding 
        '''
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self,model_output):
        '''
        take the CLS token of the last layer for input sentences as sentence embeding 
        '''
        token_embeddings = model_output[0]
        cls_embeddings = token_embeddings[:,0] ## take the first token (CLS token) as embeding 
        return cls_embeddings

    def weightedmean_pooling(self,model_output,attention_mask):
        raise('Not yet implemented')
        return None

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_name = 'bert-base-cased'
emb_model = AutoModelForSentenceEmbedding(model_name=m_name,tokenizer_name=m_name)
emb_model.to(device)
#%%
optimizer = AdamW(params=emb_model.parameters(), lr=2e-5, correct_bias=True)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)
    


batch = [['this is a test sentence','this is a similar test sentence','complete contradictory to anything else'],
        ['this is a test sentence','this is a similar test sentence','complete contradictory to anything else']]

if len(batch[0])>2:
    print('use constrative loss with hard negative')
    text1 = emb_model.tokenizer([b[0] for b in batch], return_tensors="pt", truncation=True, padding="max_length") #max_length=args.max_length, 
    text2 = emb_model.tokenizer([b[1] for b in batch], return_tensors="pt", truncation=True, padding="max_length") 
    text3 = emb_model.tokenizer([b[2] for b in batch], return_tensors="pt", truncation=True, padding="max_length") 
    
    embeddings_a  = emb_model(**text1.to(device))
    embeddings_b1 = emb_model(**text2.to(device))
    embeddings_b2 = emb_model(**text3.to(device))
#%%
embeddings_b = torch.cat([embeddings_b1,embeddings_b2],dim=0) ## append two embedings 
scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) ## a * b^T


#%%

if __name__ == "__main__":

# #%%
# m_name = 'bert-base-cased'
# M = AutoModelForSentenceEmbedding(model_name=m_name,tokenizer_name=m_name)
# # %%
# test_sentences = ['this is just a test sentence','text # 2']
# embs = M.get_embedings(test_sentences)
# #%%



