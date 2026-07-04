
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm

import json
import re


class BPETokenizer:

    def __init__(

        self,

        vocabulary_size=32768,

        special_tokens=None

    ):

        if special_tokens is None:

            special_tokens = [
                "<PAD>",
                "<UNK>",
                "<BOS>",
                "<EOS>"
            ]

        self.vocabulary_size = vocabulary_size

        self.special_tokens = special_tokens

        self.word_frequencies = Counter()

        self.vocabulary = {}

        self.inverse_vocabulary = {}

        self.merges = []

        self.is_trained = False


if __name__ == "__main__":

    print("=" * 60)
    print("Raphtaliya BPE Tokenizer")
    print("Version : Final")
    print("=" * 60)
