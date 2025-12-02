import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer

from config import config


def print_learable_params_name(atten: nn.Module):
    for name, param in atten.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def text_encoder(text: str):
    tokenizer_name = config.hf_tokenizer_model_name
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded_text = tokenizer.encode(text, return_tensors='pt')

    return encoded_text

def text_decoder(tokens: torch.Tensor):
    tokenizer_name = config.hf_tokenizer_model_name
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded_text = tokenizer.decode(tokens)
    return encoded_text