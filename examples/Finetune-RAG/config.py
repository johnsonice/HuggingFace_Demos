import os 

## global folder path 
# data_folder2= "/Users/huang/Dev/projects/All_Data/HuggingFace"
# data_folder3= "/data/chuang/Language_Model_Training_Data"
cache_dir = "/data/hg_cache"
sotrage_folder= "/data/LLM_DATA/Fund_docs"
data_folder = os.path.join(sotrage_folder,'training_data') 
model_folder = os.path.join(sotrage_folder,'models')  
## default model params   
default_model_checkpoint = "BAAI/bge-large-en-v1.5"  #"BAAI/bge-base-en-v1.5" 
## other params 
RANDOM_SEED = 42