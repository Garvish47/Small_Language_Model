
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            B, T, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim, num_heads, dropout
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        max_seq_len=128,
        dropout=0.1
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim, num_heads, ff_dim, dropout
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: input embedding = output projection weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError("Sequence exceeds maximum context length")

        positions = torch.arange(T, device=input_ids.device)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        x = self.blocks(x)
        x = self.final_ln(x)

        logits = self.lm_head(x)
        return logits
