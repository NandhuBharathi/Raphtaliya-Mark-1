
import math
import torch
import torch.nn as nn


class Evaluator:

    def __init__(
        self,
        model,
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits = self.model(input_ids)

            vocab_size = logits.size(-1)

            loss = self.loss_fn(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )

            total_loss += loss.item()
            total_batches += 1

        average_loss = (
            total_loss / total_batches
            if total_batches > 0 else 0.0
        )

        perplexity = math.exp(
            average_loss
        ) if average_loss < 20 else float("inf")

        return {
            "loss": round(average_loss, 4),
            "perplexity": round(perplexity, 4)
            if perplexity != float("inf")
            else "Infinity",
            "batches": total_batches,
            "device": str(self.device)
        }

    def model_parameters(self):

        return sum(
            p.numel()
            for p in self.model.parameters()
        )


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Evaluator",
        version="V2.0"
    )
