### train customized

"""
Follow : 
- https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/train_script.py
- https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/modeling.py
"""
#%%
import sys
sys.path.insert(0,'../../libs')
from arguments import train_args
from data_utils import load_train_eval_dataset
from SBERT_HF_utils import SentenceTransformersCollator
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from models import AutoModelForSentenceEmbedding

class EmbeddingModel(AutoModelForSentenceEmbedding):
    def forward(self, **kwargs):
        model_output = self.model(input_ids=kwargs.get('input_ids'),
                                attention_mask= kwargs.get('attention_mask')
                                )
        embeddings = self._mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
#%%
if __name__ == "__main__":
    args = train_args([])
    train_ds,eval_ds = load_train_eval_dataset(args.train_file,args.eval_file)
    tokenizer =  AutoTokenizer.from_pretrained(args.model_checkpoint,cache_folder =args.cache_dir)
    ## let's just tokenize it on the fly using data collator
    text_columns = ['question','context']
    data_collator = SentenceTransformersCollator(
         tokenizer=tokenizer,
         text_columns=text_columns
    )
    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size,collate_fn=data_collator)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size,collate_fn=data_collator)
    # %%
    model = EmbeddingModel(args.model_checkpoint,cache_dir=args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    # setup warmup for first ~10% of steps
    epochs=3
    total_steps = int(len(train_ds)*epochs / batch_size)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps-warmup_steps
    )
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)
    #%%
    for epoch in range(epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # zero all gradients on each new step
            optim.zero_grad()
            anchor_ids = batch['question_input_ids'].to(device)
            anchor_mask = batch['question_attention_mask'].to(device)
            pos_ids = batch['context_input_ids'].to(device)
            pos_mask = batch['context_attention_mask'].to(device)

            anchor_emb = model(input_ids=anchor_ids,attention_mask=anchor_mask)
            pos_emb = model(input_ids=pos_ids,attention_mask=pos_mask)
            scores = torch.stack([cos_sim(a_i.reshape(1, a_i.shape[0]), pos_emb) for a_i in anchor_emb])
            # get label(s) - we could define this before if confident of consistent batch sizes
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
            # and now calculate the loss
            loss = loss_func(scores*scale, labels)
            # using loss, calculate gradients and then optimize
            loss.backward()
                    # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


# #%%
#     #%%
#     test_ins = tokenizer(train_ds[:8]['question'],truncation=True,return_tensors='pt',padding=True)
    
#     emb = model(input_ids=test_ins['input_ids'].to(device),
#                 attention_mask=test_ins['attention_mask'].to(device),
#                 question=train_ds[:8]['question'])
#     # %%
#     for batch in train_loader:
#         b = batch
#         break
        


    # %%
