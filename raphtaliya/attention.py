
import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):

        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

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

        scores = scores / math.sqrt(
            self.head_dim
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

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
