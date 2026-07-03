
from pathlib import Path


class Dataset:

    def __init__(self, folder):

        self.folder = Path(folder)

    def load(self):

        texts = []

        for file in sorted(self.folder.rglob("*.txt")):

            with open(file, "r", encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if line:
                        texts.append(line)

        return texts

    def __len__(self):

        return len(self.load())
