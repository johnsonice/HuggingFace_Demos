import torch
import os

from llmtuner import create_ui
os.environ['CUDA_VISIBLE_DEVICES'] = '0' ## to run gui for now, have to set only single GPU
create_ui().queue().launch(share=True)