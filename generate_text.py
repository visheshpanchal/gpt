import torch

from model import GPT
from config import config
import torch.nn.functional as F

from utils import text_decoder, text_encoder


def generate_text(tokens: torch.Tensor):
    model = GPT(config)

    # New max tokens
    context_length = config.context_length
    new_max_tokens = context_length - len(tokens[0])

    while new_max_tokens > 0:
        print(new_max_tokens)
        with torch.no_grad():
            logits = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ids_next = torch.argmax(probs, dim=-1, keepdim=True)
            tokens = torch.cat((tokens, ids_next), dim=1)
        new_max_tokens -= 1

    return tokens[0]


if __name__ == '__main__':
    tokens = text_encoder('Hello World')
    token_ids = generate_text(tokens)
    text = text_decoder(token_ids)
    print(text)
