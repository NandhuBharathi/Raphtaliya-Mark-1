
from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.trainer import Trainer

import torch

VOCAB_SIZE = 100
EMBED_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 1024
NUM_LAYERS = 4
MAX_SEQUENCE_LENGTH = 512

model = RaphtaliyaMark1(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)

trainer = Trainer(model)

for epoch in range(10):

    inputs = torch.randint(0, VOCAB_SIZE, (8, 32))
    targets = torch.randint(0, VOCAB_SIZE, (8, 32))

    loss = trainer.train_step(inputs, targets)

    print(f"Epoch {epoch + 1} | Loss: {loss:.4f}")
