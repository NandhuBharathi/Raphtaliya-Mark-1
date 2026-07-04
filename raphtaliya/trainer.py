
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:

    def __init__(
        self,
        model,
        learning_rate=3e-4,
        gradient_clip=1.0,
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
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

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

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

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

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
            "average_loss": round(average, 4),
            "device": str(self.device)
        }

    def reset_statistics(self):

        self.training_steps = 0
        self.loss_history.clear()

    def model_parameters(self):

        return sum(
            p.numel()
            for p in self.model.parameters()
        )

    def train_mode(self):

        self.model.train()

    def eval_mode(self):

        self.model.eval()


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Trainer",
        version="V3.0"
    )
