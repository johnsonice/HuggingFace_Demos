
#%%
import sys,os,gzip,csv,math
sys.path.insert(0,'..')
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
from eval import process_sts
from data_utils import get_sbert_nli_data,add_to_samples,nli2triplet_sets,transform2triplet_inputexamples

#%%
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#%%
def prepare_anli_training_examples(data_path='/home/chuang/Dev/DATA/raw_data'):
    # Read the AllNLI.tsv.gz file and create the training dataset
    #Check if dataset exsist. If not, download and extract  it
    nli_dataset_path = os.path.join(data_path,'AllNLI.tsv.gz')
    get_sbert_nli_data(nli_dataset_path)
    # read and transform data 
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        train_data = nli2triplet_sets(reader,filter_key='split',split='train')
    print('Print one dict item')
    print(next(iter(train_data.items())))
    # transform into triples training examples 
    train_samples = transform2triplet_inputexamples(train_data)
    return train_samples

def get_sts_evalator():
    ## get data and setup evaluator 
    sts_samples = process_sts()
    sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        sts_samples, write_csv=False
    )
    return sts_evaluator
#%%

if __name__ == "__main__":
    
    model_name = 'distilroberta-base'
    train_batch_size = 128          #The larger you select this, the better the results (usually). But it requires more GPU memory
    max_seq_length = 75
    num_epochs = 1
    model_save_path = '/home/chuang/Dev/DATA/Model/training_nli_v2_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = '/home/chuang/Dev/DATA/raw_data'

    logging.info("Read AllNLI train dataset")
    train_samples = prepare_anli_training_examples(data_path)
    logging.info("Train samples: {}".format(len(train_samples)))
    #%%
    # Special data loader that avoid duplicates within a batch
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    # Here we define our SentenceTransformer model
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Our training loss
    logging.info('User multipleNegativesRankingLoss')
    train_loss = losses.MultipleNegativesRankingLoss(model)
    #%%
    # set up evaluator for sbert 
    dev_evaluator = get_sts_evalator()
    #%%
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=int(len(train_dataloader)*0.05),
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            use_amp=False          #Set to True, if your GPU supports FP16 operations
            )


