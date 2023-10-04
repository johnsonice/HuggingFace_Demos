#%%
import os,sys,openai
import pandas as pd 
from data_utils import get_completion
sys.path.insert(0,'../../libs')
from utils import load_json
from prompts import gen_q_basic,gen_q_fewshot,gen_q_cot
import tqdm,re
# from llama_index import Document
# from llama_index.node_parser import SimpleNodeParser
# from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo
# from llama_index.llms import OpenAI
# from llama_index.finetuning import generate_qa_embedding_pairs

key = load_json('../../../openai_key.json') 
os.environ['OPENAI_API_KEY'] = key['ChatGPT1']['API_KEY']
openai.api_key  = os.getenv('OPENAI_API_KEY')

def load_process_aiv(file_path,keep_n=None):
    aiv_df = pd.read_csv(file_path)
    aiv_df = aiv_df[(aiv_df['word_count']>80) & (aiv_df['word_count']<400)]
    aiv_df = aiv_df[2000:]
    
    if keep_n:
        aiv_df = aiv_df.sample(n=keep_n)
        
    return aiv_df

def compare_prompts_results(prompts:list,context_info,n_questions):
    for p in prompts:
        print('\n\nPrompt used: \n{} \n'.format(p['des']))
        print(get_completion(prompt=p['Human'].format(context_str=context_info,
                                                        num_questions_per_chunk=n_questions),
                        sys_msg=p['System'],
                        model='gpt-3.5-turbo'))
        
    return None


#%%
if __name__ == "__main__":
    
    ## load raw context data 
    data_folder = '/data/LLM_DATA/Fund_docs'
    raw_aiv_file = os.path.join(data_folder,'aiv.20230820.csv')
    raw_program_file = os.path.join(data_folder,'program.20230820.csv')
    keep_n = 50
    aiv_df = load_process_aiv(raw_aiv_file,keep_n=keep_n)
    program_df = load_process_aiv(raw_program_file,keep_n=keep_n)

    #%%
    ## simple test question generation using different prompt
    context_info = 'To accelerate growth while maintaining macroeconomic stability, policies should continue to focus on: (i) accelerating efforts to attract investment and raise the economy’s potential by improving the business environment, reforming the large state-owned enterprise (SOE) sector, developing a market for agricultural land, and tackling corruption, which remains a key challenge; (ii) ensuring fiscal sustainability through fiscal consolidation, supported by pension reform, more efficient public spending, and a more equitable and growth-friendly tax system; (iii) further reducing inflation and rebuilding reserves; and (iv) repairing viable banks and reviving sound bank lending.'
    n_questions = 5
    compare_prompts_results([gen_q_basic,gen_q_fewshot,gen_q_cot],context_info,n_questions)
    
    #%%
    prompt_template = gen_q_basic
    n_questions = 3
    res = []
    for context_info in tqdm.tqdm(aiv_df['par'][:2]):
        response = get_completion(prompt=prompt_template['Human'].format(context_str=context_info,
                                                        num_questions_per_chunk=n_questions),
                                    sys_msg=prompt_template['System'],
                                    model='gpt-3.5-turbo')
        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]
        add_obs = [(q,context_info) for q in questions]
        res.extend(add_obs)
    #%%


# def generate_q_with_llamaindex():
#     '''
#     This is not very flexible for real world usages as we need to do quality check probably in excel at some point
#     '''
#     ## generate data for aiv
#     ## follow https://gpt-index.readthedocs.io/en/v0.7.7/end_to_end_tutorials/usage_pattern.html
#     ## https://github.com/run-llama/finetune-embedding/blob/main/generate_dataset.ipynb
#     text_list = aiv_df['par'].tolist()
#     #documents = [Document(text=t) for t in text_list]
#     nodes = [TextNode(text=t) for t in text_list]
#     train_nodes = nodes[:40]
#     val_nodes = nodes[40:50]
#     llm = OpenAI(
#     api_key=os.environ['OPENAI_API_KEY'],
#     model='gpt-3.5-turbo',
#     temperature=0.0
#     )   
#     qa_generate_prompt = gen_q_basic['Human']
#     train_dataset = generate_qa_embedding_pairs(train_nodes,
#                                           llm=llm,
#                                           qa_generate_prompt_tmpl=qa_generate_prompt,
#                                           num_questions_per_chunk=3)

#     val_dataset = generate_qa_embedding_pairs(val_nodes,
#                                             llm=llm,
#                                             qa_generate_prompt_tmpl=qa_generate_prompt,
#                                             num_questions_per_chunk=3)
#     #%%
#     out_path = os.path.join(data_folder,"aiv_train_dataset.json")
#     train_dataset.save_json(out_path)
#     out_path = os.path.join(data_folder,"aiv_val_dataset.json")
#     val_dataset.save_json(out_path)
    
#     return None
# %%
