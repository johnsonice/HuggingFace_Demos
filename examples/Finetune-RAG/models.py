
from dataclasses import dataclass
#from transformers.optimization import get_linear_schedule_with_warmup
from typing import Optional
import torch
from torch import nn, Tensor
from transformers import (
    AutoModel,
)
from transformers.file_utils import ModelOutput


@dataclass
class EncoderOutput(ModelOutput):
    anchor_emb: Optional[Tensor] = None
    pos_emb: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, 
                 model_name, 
                 sentence_pooling_method: str = 'mean',
                 cache_dir=None,
                 normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name,
                                               cache_dir=cache_dir)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normalize = normalize
        self.sentence_pooling_method = sentence_pooling_method
        self.cos_sim = torch.nn.CosineSimilarity()
    
    def forward(self,**kwargs):
        ## this part it hard coded for now, may want to chagne latter 
        anchor_ids = kwargs.get('question_input_ids')
        anchor_mask = kwargs.get('question_attention_mask')
        pos_ids = kwargs.get('context_input_ids')
        pos_mask = kwargs.get('context_attention_mask')
        
        anchor_emb = self.get_embedding(input_ids=anchor_ids,attention_mask=anchor_mask)
        pos_emb = self.get_embedding(input_ids=pos_ids,attention_mask=pos_mask)        
        
        if self.training:
            scores = self.compute_similarity(anchor_emb, pos_emb)
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
            # and now calculate the loss
            loss = self.compute_loss(scores*20, labels) ## just simpley scale it be 20 for now; will replace with temp latter
        else:
            scores = self.compute_similarity(anchor_emb, pos_emb)
            loss = None
        
        return EncoderOutput(
            loss=loss,
            scores=scores,
            anchor_emb=anchor_emb,
            pos_emb=pos_emb,
        )
    
    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def get_embedding(self,**kwargs):
        model_output = self.model(input_ids=kwargs.get('input_ids'),
                                  attention_mask= kwargs.get('attention_mask')
                                  )
        if self.sentence_pooling_method == "mean":
            embeddings = self._mean_pooling(model_output, kwargs['attention_mask'])
        if self.sentence_pooling_method == "cls":
            embeddings = self._cls_pooling(model_output)
        
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity(self, anchor_emb, pos_emb):
        scores = torch.stack([self.cos_sim(a_i.reshape(1, a_i.shape[0]), pos_emb) for a_i in anchor_emb])
        return scores 
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output):
        return model_output[0,0,:]