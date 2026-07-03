
import torch.nn as nn

from raphtaliya.embedding import TokenEmbedding
from raphtaliya.position import PositionalEmbedding
from raphtaliya.layers import TransformerBlock


class RaphtaliyaMark1(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_sequence_length
    ):

        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        x = self.token_embedding(token_ids)

        x = self.position_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.lm_head(x)
