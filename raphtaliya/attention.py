
import math
import torch
import torch.nn as nn

from raphtaliya.config import DROPOUT

from raphtaliya.config import DROPOUT


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=DROPOUT
    ):

        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.output = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attention_mask=None
    ):

        batch_size, sequence_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        if attention_mask is not None:

            scores = scores.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):

            context = torch.nn.functional.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=causal_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )

        else:

            attention = torch.softmax(
                scores,
                dim=-1
            )

            attention = self.dropout(attention)

            context = torch.matmul(
                attention,
                V
            )

        context = context.transpose(
            1,
            2
        ).contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        return self.output(context)
