### Example Mistral 7B

#%%
import os
from threading import Thread
from typing import Iterator

import gradio as gr
#import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import transformers
#%%

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 128
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

my_api_token = 'xxx'

#%%
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")#, token=my_api_token) #, token=my_api_token
    # to suppress warning: The attention mask and the pad token id were not set. 
    # As a consequence, you may observe unexpected behavior. 
    # Please pass your input's `attention_mask` to obtain reliable results. 
    # Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)#, token=my_api_token)
    tokenizer.use_default_system_prompt = False

#%%
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.5,
) -> Iterator[str]:
    
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    ## format caht history #https://huggingface.co/docs/transformers/main/en/chat_templating
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt",add_generation_prompt=True) 
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    ## set up hf streamer
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        eos_token_id=[model.generation_config.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
        
# #%%
# messages = []
# system_prompt = 'Your name is Llama 3. You are a useful AI Chatbot that responds to user questions! '

# res = generate("How are you today",system_prompt=None,chat_history=messages)
# for t in res:
#         print(f'\r{t}', end='',flush=True)

#%%
chat_interface = gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)

DESCRIPTION = """\
# Meta-LLAMA3 Chat

This Space demonstrates [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

"""

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)