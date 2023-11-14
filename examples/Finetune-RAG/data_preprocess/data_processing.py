#%%
## data sampling 
import os 
import pandas as pd
from arguments import train_args
from sklearn.model_selection import train_test_split
#%%
if __name__ == "__main__":
    args = train_args([])
    #%% Split GPT generated data into train and test split 
    train_df = pd.read_excel(args.train_file)
    gen_train_df,gen_test_df = train_test_split(train_df, test_size=0.3, random_state=42)
    gen_train_df.to_excel(os.path.join(args.data_folder,'gpt_train_split.xlsx'),index=False)
    gen_test_df.to_excel(os.path.join(args.data_folder,'gpt_test_split.xlsx'),index=False)
    #%%
    eval_df = pd.read_excel(args.eval_file)
    gen_train_df,gen_test_df = train_test_split(eval_df, test_size=0.5, random_state=42)
    gen_train_df.to_excel(os.path.join(args.data_folder,'human_train_split.xlsx'),index=False)
    gen_test_df.to_excel(os.path.join(args.data_folder,'human_test_split.xlsx'),index=False)
# %%
