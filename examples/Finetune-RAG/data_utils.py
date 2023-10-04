## data process utils

import openai
import os

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