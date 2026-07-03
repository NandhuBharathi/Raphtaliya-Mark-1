
import torch
import torch.nn as nn

from raphtaliya.attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.feedforward = FeedForward(
            embedding_dim,
            hidden_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        x = self.norm1(
            x + self.attention(x)
        )

        x = self.norm2(
            x + self.feedforward(x)
        )

        return x
