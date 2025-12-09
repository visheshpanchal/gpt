import torch


class Config:
    ## Model Config
    embed_dim = 768
    n_heads = 12
    context_length = 1024
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout = 0.1
    qkv_bias = False
    vocab_size = 50257
    n_transformer = 12
    hf_tokenizer_model_name = 'openai-community/gpt2'
    no_of_epoch = 10
    learning_rate = 1e-4

    class DataConfig:
        data_dir = 'Salesforce/wikitext'
        save_data_dir = 'llmdata/data'
        save_token_dir = 'llmdata/tokens'
        model_name_from_hf = 'openai-community/gpt2'

        root_is_home = True
        data_from_hf = True

        max_length = 256 # Length of context
        stride = 128
        max_tokens = 1000_000_0 # Fetch max tokens from dataset

    _data = DataConfig()

    @property
    def data(self):
        return self._data


config = Config()