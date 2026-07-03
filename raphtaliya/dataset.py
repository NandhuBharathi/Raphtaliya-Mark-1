
from pathlib import Path


class Dataset:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def load(self):

        books = []

        for file in sorted(self.dataset_path.glob("*.txt")):

            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                books.append(f.read())

        return books
