import ast
import re

def parst_question_list(response):
    result = str(response).strip().split("\n")
    questions = [
        re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
    ]
    questions = [question for question in questions if len(question) > 0]

    return questions 

gen_q_basic = {'des':'Most basic questions generation prompt',
"System":""" 
You are an ecumenist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
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
You are an ecumenist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
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
You are an ecumenist at the International Monetary Fund (IMF). Your primary role is grounded in economics. 
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
# %%
