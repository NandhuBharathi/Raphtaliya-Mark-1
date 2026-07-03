
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:

    def __init__(
        self,
        model,
        learning_rate=3e-4
    ):

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    def train_step(
        self,
        input_ids,
        target_ids
    ):

        self.model.train()

        self.optimizer.zero_grad()

        logits = self.model(input_ids)

        vocab_size = logits.size(-1)

        loss = self.loss_fn(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()
