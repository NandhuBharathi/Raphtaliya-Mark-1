
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_sequence_length, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        return x + self.embedding(positions)
