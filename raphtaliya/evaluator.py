import math
import torch
import torch.nn as nn


class Evaluator:

    def __init__(self, model):

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for input_ids, target_ids in dataloader:

            logits = self.model(input_ids)

            vocab_size = logits.size(-1)

            loss = self.loss_fn(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / total_batches

        perplexity = math.exp(average_loss)

        return {
            "loss": round(average_loss, 4),
            "perplexity": round(perplexity, 4),
            "batches": total_batches
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Evaluator",
        version="V1.0"
    )
