from pathlib import Path
from datetime import datetime
import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, filename):

        return self.directory / filename

    def save(
        self,
        model,
        optimizer=None,
        epoch=0,
        loss=0.0,
        filename="latest.pt",
        metadata=None
    ):

        checkpoint = {
            "version": "2.0",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {}
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        path = self._path(filename)

        torch.save(checkpoint, path)

        return path

    def load(
        self,
        model,
        optimizer=None,
        filename="latest.pt",
        device="cpu"
    ):

        path = self._path(filename)

        checkpoint = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(device)
        model.eval()

        if (
            optimizer is not None
            and
            "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint

    def save_best(
        self,
        model,
        optimizer=None,
        epoch=0,
        loss=0.0,
        metadata=None
    ):

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            filename="best.pt",
            metadata=metadata
        )

    def save_latest(
        self,
        model,
        optimizer=None,
        epoch=0,
        loss=0.0,
        metadata=None
    ):

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            filename="latest.pt",
            metadata=metadata
        )

    def load_best(
        self,
        model,
        optimizer=None,
        device="cpu"
    ):

        return self.load(
            model=model,
            optimizer=optimizer,
            filename="best.pt",
            device=device
        )

    def load_latest(
        self,
        model,
        optimizer=None,
        device="cpu"
    ):

        return self.load(
            model=model,
            optimizer=optimizer,
            filename="latest.pt",
            device=device
        )

    def exists(self, filename):

        return self._path(filename).exists()

    def list_checkpoints(self):

        return sorted(
            [
                file.name
                for file in self.directory.glob("*.pt")
            ]
        )

    def delete(self, filename):

        path = self._path(filename)

        if path.exists():
            path.unlink()
            return True

        return False

    def checkpoint_info(self, filename):

        path = self._path(filename)

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

        return {
            "version": checkpoint.get("version"),
            "epoch": checkpoint.get("epoch"),
            "loss": checkpoint.get("loss"),
            "created_at": checkpoint.get("created_at"),
            "metadata": checkpoint.get("metadata")
        }

    def verify(self, filename):

        try:

            checkpoint = torch.load(
                self._path(filename),
                map_location="cpu"
            )

            return (
                "model_state_dict" in checkpoint
                and
                "epoch" in checkpoint
            )

        except Exception:

            return False


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Checkpoint",
        version="V2.0"
    )
