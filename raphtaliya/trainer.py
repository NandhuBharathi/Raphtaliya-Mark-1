import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:

    def __init__(
        self,
        model,
        learning_rate=3e-4,
        gradient_clip=1.0
    ):

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        self.gradient_clip = gradient_clip

        self.training_steps = 0

        self.loss_history = []

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

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )

        self.optimizer.step()

        loss = loss.item()

        self.training_steps += 1

        self.loss_history.append(loss)

        return loss

    @torch.no_grad()
    def evaluate(
        self,
        input_ids,
        target_ids
    ):

        self.model.eval()

        logits = self.model(input_ids)

        vocab_size = logits.size(-1)

        loss = self.loss_fn(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        return loss.item()

    def learning_rate(self):

        return self.optimizer.param_groups[0]["lr"]

    def statistics(self):

        average = 0.0

        if self.loss_history:
            average = sum(self.loss_history) / len(self.loss_history)

        return {
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate(),
            "average_loss": round(average, 4)
        }

    def reset_statistics(self):

        self.training_steps = 0
        self.loss_history.clear()


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Trainer",
        version="V2.0"
    )
