
from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer=None, epoch=0, loss=0.0, filename="latest.pt"):

        path = self.directory / filename

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict()
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

        return path

    def load(self, model, optimizer=None, filename="latest.pt"):

        path = self.directory / filename

        checkpoint = torch.load(path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint
