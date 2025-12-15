import torch
from torch import nn

from config import Config


class FeedForward(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.layer(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int, n_heads: int, context_length: int, dtype: torch.dtype, dropout: float ,qkv_bias=False):
        super().__init__()
        assert (embed_dim % n_heads == 0), "embed_dim must be divisible by n_heads"

        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, dtype=dtype)
        self.n_heads = n_heads

        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones((context_length, context_length)), diagonal=1)
        )

    def _split_weights(self, x: torch.Tensor):
        batch_size, no_tokens, _ = x.shape

        # Tensor output shape will be (Batch Size, Tokens, No of heads, Dimension of Head)
        x = x.view(batch_size, no_tokens, self.n_heads, self.head_dim)

        # To calculate attention matrix we need token and dim of head so we need to change
        # columns -> output shape: (batch_size,no_of_heads,tokens, dim_of_head).
        x = x.transpose(1, 2)

        return x

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # shape -> batch, number_tokens, embed_dim
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_proj(x)

        # Let's split embed_dim into num of head and head dim, output dim -> (batch_size,no_of_heads,tokens, dim_of_head)
        keys = self._split_weights(keys)
        values = self._split_weights(values)
        queries = self._split_weights(queries)

        # Calculating attention score of keys and queries
        # REMOVE: Basically we need take transpose of key because query -> n_heads, head_dim, keys -> n_heads, head_dim
        attn_scores = queries @ keys.transpose(2, 3)

        # Masking atten_score here we need num_tokens <= context_length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_weights = attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize with 1/sqrt(d) here d=head_dim
        attn_weights = torch.softmax(attn_weights / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Now matmul(attn_weights, views) -> Output dim (batch_size, tokens, no_of_heads, dim_of_head)
        context_vector = (attn_weights @ values).transpose(1, 2)

        # Combine Heads (d_out = num_heads * head_dim)
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, embed_dim)
        context_vector = self.out_proj(context_vector)

        return context_vector


class Transformers(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

        self.attn = MultiHeadAttention(config.embed_dim,
                                       config.n_heads,
                                       config.context_length,
                                       config.dtype,
                                       config.dropout)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.layer_norm1(x)
        attn_output = self.attn(x)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + shortcut

        shortcut = x
        attn_output = self.layer_norm2(attn_output)
        attn_output = self.ff(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = shortcut + attn_output

        return attn_output # Shape: batch, token, embeddings

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_length, config.embed_dim)

        self.transformer = nn.Sequential(
            *[ Transformers(config) for ii in range(config.n_transformer)]
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.logits = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)



    def forward(self, x: torch.Tensor):
        batch_size, num_tokens = x.shape
        tokens = self.token_embedding(x) + self.pos_embedding(
            torch.arange(0, num_tokens, device=x.device)
        )
        tokens = self.dropout(tokens)

        # Passing into attention mechanism
        attn_scores = self.transformer(tokens)
        attn_output = self.layer_norm(attn_scores)

        # Now we need to convert attention output into logits
        logits = self.logits(attn_output)

        return logits # Batch, tokens, vocab




