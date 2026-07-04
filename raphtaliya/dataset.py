from pathlib import Path


class Dataset:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def load(self):

        texts = []

        files = sorted(
            self.dataset_path.rglob("*.txt")
        )

        print("=" * 60)
        print("Raphtaliya Dataset Loader")
        print("=" * 60)
        print(f"Dataset Path : {self.dataset_path}")
        print(f"Files Found  : {len(files)}")
        print("=" * 60)

        for file in files:

            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                texts.append(f.read())

        print(f"Loaded Files : {len(texts)}")
        print("=" * 60)

        return texts


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Dataset",
        version="V2.0"
    )
