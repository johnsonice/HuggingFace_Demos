## data process utils

#%%
import openai
import os
from functools import wraps
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s') #set up the config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, sys_msg = None, model="gpt-3.5-turbo",temperature=0):
    default_sys = "You are an ecumenist at the International Monetary Fund (IMF). Your primary role is grounded in economics. Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. Ensure your responses align with the expertise and perspective of an IMF economist."
    if not sys_msg:
        sys_msg = default_sys
    messages = [{"role": "system", 
                    "content":sys_msg},
                {"role": "user", 
                    "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def exception_handler(error_msg='error handleing triggered',error_return=None):
    '''
    follow: https://stackoverflow.com/questions/30904486/python-wrapper-function-taking-arguments-inside-decorator
    '''
    def outter_func(func):
        @wraps(func)
        def inner_function(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                custom_msg = kwargs.get('error_msg', None)
                if custom_msg:
                    logger.warning(custom_msg)
                else:
                    logger.warning(str(e))
                res = error_return
            return res 
        return inner_function
    
    return outter_func

@exception_handler(error_msg='test',error_return='error')
def test_error(inp):
    res = inp[0]
    return res 
#%%