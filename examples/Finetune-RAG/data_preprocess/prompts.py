import ast
import re

#%%
def parst_question_list(response):
    result = str(response).strip().split("\n")
    questions = [
        re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
    ]
    questions = [question for question in questions if len(question) > 0]

    return questions 

def clean_text(response):
    
    return str(response).strip()

def extract_dict(response):
    return ast.literal_eval(response)

def extract_list(response):
    return ast.literal_eval(response)

gen_q_basic = {'des':'Most basic questions generation prompt',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.generate only questions based on the below query.

You are an IMF Economist. Your task is to setup {num_questions_per_chunk} questions that an IMF economist is likely going to ask based on the context provided. 
Questions should be very short (less than 15 words) and diverse in nature across the document. Restrict the questions to the context information provided. 
""",
'parsing_func':parst_question_list}


gen_q_fewshot = {'des':'Few-Shot questions generation prompt',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.generate only questions based on the below query.

Few-shot examples below:

---------------------
Which structural reforms could authorities implement to increase tax revenues?
How can the authorities use fiscal policy in response to an economic crisis?
What are the main barriers to financial inclusion?
What are the key options to restructure NPLs?
How can authorities anchor fiscal policy and support macroeconomic stability over the medium-term?
What are the key pillars of a Public Finance Management reform plan?
Which policies should countries adopt to help improve human capital? 
what could authorities do to prepare for future liquidity shocks?
What would be the effects of an increase in minimum wages on economic informality?
---------------------

You are an IMF Economist. Your task is to setup {num_questions_per_chunk} questions that an IMF economist is likely going to ask based on the context provided. 
The questions should be very short (less than 15 words) and diverse in nature across the document. Restrict the questions to the context information provided.  
You can follow the language style provided in the few-shot examples to generate your questions. Restrict the questions to the context information provided.
""",
'parsing_func':parst_question_list}

### to be compatable with llama index, COT export format are generated questions only, which will limit the effect of COT
### need to redo the generate_qa_embedding_pairs function if want to use real COT
### https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/common.py#L60

#%%
gen_q_cot = {'des':'Chain of Thought questions generation prompt',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.generate only questions based on the below query.

You are an IMF Economist. Your task is to setup {num_questions_per_chunk} questions that an IMF economist is likely going to ask based on the context provided. 
You can break down the task into several steps:
1. You can summarize context into several points that are helpful to IMF economists.
2. Generate questions according to those key points.
3. Edit the {num_questions_per_chunk} generated questions. Make sure they are very short, around 15 words, and diverse in nature across the document. Restrict the questions to the context information provided.
Make sure you return results as a python dictionary with following keys: "KeyPoints", "Generated_Questions" and "Edited_Question"
Return results from each step as a list of items under each key.
""",
'parsing_func':ast.literal_eval}

###################################################################
############# Prompts for sumamry generation ######################
###################################################################

gen_query_basic = {'des':'Basic prompt to transform question to sub queries',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Your main task is to transform the complex user question into a concise query for information retrieval and return it in a python dictionary.
Please follow the following steps:
1. Transform the complex user question into a concise and direct query for information retrieval. Feel free to break it into several sub-queires when needed.
2. Please also extract and list keywords from user's question related to the 'Topic', 'Goal'.
3. Please make sure your results are in a python dictionary with keys 'Query', 'Topic', 'Goal'. Please only return the python dictionary only an nothing else.

User question is below:
----------
{user_question}
----------

Please return python dictionary below:
""",
'parsing_func':extract_dict}
#########
gen_summary_basic = {'des':'Basic summarization generation prompt',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.

---------------------
{context_str}
---------------------

Read the provided context carefully and create an extractive summary.
Please follow the specific instructions below:
1. Your summary should consist of paragraphs or key sentences extracted directly from the context. 
2. Aim to construct a coherent narrative that remains true to the source material. 
3. Your final text content should not exceed three short paragraphs and should be no more than 400 words in total. 

Please generate response below:
""",
'parsing_func':clean_text}

########
gen_summary_on_query = {'des':'Extractive summarization based on a query',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.
---------------------
{context_str}
---------------------

User Query is below:
---------------------
{query}
---------------------

Read the provided context carefully and perform the task following these steps and return a python list object:
1. extract the most relevant information that responds to the query provided above.
2. Please make sure that your response should consist of direct quotes of paragraphs or key sentences extracted directly from the context. 
3. Your final answer should not exceed 4 paragraphs and should be no more than 300 words in total. 
4. Please make sure to format you answer as a python list with most relevant paragraphs. Do not return anything else. 

Please return python list below:
""",
'parsing_func':extract_list}

########
gen_summary_on_keywords = {'des':'Extractive summarization based on a list of keywords',
"System":""" 
You are an economist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. 
Ensure your responses align with the expertise and perspective of an IMF economist.
""", 
"Human" : """
Context information is below.
---------------------
{context_str}
---------------------

Query keywords are below:
---------------------
{keywords}
---------------------

Read the provided context carefully and perform the task following these steps and return a python list object:
1. extract the most relevent information that responds to the query keywords provided above.
2. Please make sure that your response should consist of direct quotes of paragraphs or key sentences extracted directly from the context. 
3. Your final answer should not exceed 4 paragraphs and should be no more than 300 words in total. 
4. Please make sure to format you answer as a python list with most relevant paragraphs. Do not return anything else. 

Please return python list below:
""",
'parsing_func':extract_list}