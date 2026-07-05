import os
import json

from raphtaliya.dataset import Dataset
from raphtaliya.document_splitter import DocumentSplitter
from raphtaliya.cleaner import TextCleaner
from raphtaliya.sentence_splitter import SentenceSplitter
from raphtaliya.tokenizer import RaphtaliyaTokenizer
from raphtaliya.sequence import SequenceBuilder
from raphtaliya.padding import Padder

from raphtaliya.config import (
    SEQUENCE_LENGTH,
    TOKENIZER_PATH
)


def load_books(dataset_path):

    dataset = Dataset(dataset_path)

    books = dataset.load()

    splitter = DocumentSplitter()
    cleaner = TextCleaner()
    sentence_splitter = SentenceSplitter()

    texts = []

    for book in books:

        story = splitter.split(book)

        story = cleaner.clean(story)

        sentences = sentence_splitter.split(story)

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        texts.extend(sentences)

    return texts


def load_jsonl(folder):

    texts = []

    if not os.path.exists(folder):
        return texts

    for file in os.listdir(folder):

        if not file.endswith(".jsonl"):
            continue

        path = os.path.join(
            folder,
            file
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                data = json.loads(line)

                prompt = data.get(
                    "prompt",
                    ""
                ).strip()

                response = data.get(
                    "response",
                    ""
                ).strip()

                if prompt:
                    texts.append(prompt)

                if response:
                    texts.append(response)

    return texts
class DataPipeline:

    def __init__(
        self,
        dataset_path,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer_path=TOKENIZER_PATH
    ):

        self.dataset_path = dataset_path

        self.sequence_length = sequence_length

        self.tokenizer_path = tokenizer_path

    def build(self):

        texts = []

        # ----------------------------------
        # Books
        # ----------------------------------

        train_folder = os.path.join(
            self.dataset_path,
            "train"
        )

        if os.path.exists(train_folder):

            texts.extend(
                load_books(train_folder)
            )

        # ----------------------------------
        # Dialogue
        # ----------------------------------

        texts.extend(
            load_jsonl(
                os.path.join(
                    self.dataset_path,
                    "dialogue"
                )
            )
        )

        # ----------------------------------
        # Coding
        # ----------------------------------

        texts.extend(
            load_jsonl(
                os.path.join(
                    self.dataset_path,
                    "coding"
                )
            )
        )

        # ----------------------------------
        # Maths
        # ----------------------------------

        texts.extend(
            load_jsonl(
                os.path.join(
                    self.dataset_path,
                    "maths"
                )
            )
        )

        # ----------------------------------
        # QA
        # ----------------------------------

        texts.extend(
            load_jsonl(
                os.path.join(
                    self.dataset_path,
                    "qa"
                )
            )
        )

texts = [
            text.strip()
            for text in texts
            if text.strip()
        ]

        texts = list(
            dict.fromkeys(texts)
        )

        tokenizer = RaphtaliyaTokenizer(
            self.tokenizer_path
        )

        builder = SequenceBuilder(
            tokenizer,
            self.sequence_length
        )

        inputs, targets = builder.build(
            texts
        )

        padder = Padder()

        inputs = padder.pad(
            inputs
        )

        targets = padder.pad(
            targets
        )

        print("=" * 60)
        print("Raphtaliya Mark-1 Data Pipeline")
        print("=" * 60)
        print(
            f"Dataset Path      : {self.dataset_path}"
        )
        print(
            f"Total Texts       : {len(texts):,}"
        )
        print(
            f"Vocabulary Size   : {tokenizer.vocab_size():,}"
        )
        print(
            f"Sequence Length   : {self.sequence_length}"
        )
        print(
            f"Training Samples  : {len(inputs):,}"
        )
        print("=" * 60)
return {
            "texts": texts,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "targets": targets
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Pipeline",
        version="V5.0"
    )
