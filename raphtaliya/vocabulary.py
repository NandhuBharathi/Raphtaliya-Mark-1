from collections import Counter
import json

from raphtaliya.normalizer import Normalizer
from raphtaliya.pretokenizer import PreTokenizer


class Vocabulary:

    def __init__(self):

        self.normalizer = Normalizer()
        self.pretokenizer = PreTokenizer()

        self.counter = Counter()

        self.word_to_id = {}
        self.id_to_word = {}

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

    def build(self, texts):

        self.counter.clear()

        for text in texts:

            text = self.normalizer.normalize(text)

            tokens = self.pretokenizer.tokenize(text)

            self.counter.update(tokens)

        self.word_to_id.clear()
        self.id_to_word.clear()

        index = 0

        for token in self.special_tokens:

            self.word_to_id[token] = index
            self.id_to_word[index] = token

            index += 1

        for token, _ in self.counter.most_common():

            if token not in self.word_to_id:

                self.word_to_id[token] = index
                self.id_to_word[index] = token

                index += 1

    def get_id(self, token):

        return self.word_to_id.get(
            token,
            self.word_to_id["<UNK>"]
        )

    def get_word(self, token_id):

        return self.id_to_word.get(
            token_id,
            "<UNK>"
        )

    def size(self):

        return len(self.word_to_id)

    def contains(self, token):

        return token in self.word_to_id

    def most_common(self, n=20):

        return self.counter.most_common(n)

    def statistics(self):

        return {
            "vocabulary_size": self.size(),
            "unique_tokens": len(self.counter),
            "special_tokens": self.special_tokens
        }

    def save(self, filename):

        data = {
            "word_to_id": self.word_to_id,
            "special_tokens": self.special_tokens
        }

        with open(filename, "w", encoding="utf-8") as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4
            )

    def load(self, filename):

        with open(filename, "r", encoding="utf-8") as file:

            data = json.load(file)

        self.word_to_id = data["word_to_id"]

        self.special_tokens = data["special_tokens"]

        self.id_to_word = {
            int(index): token
            for token, index in self.word_to_id.items()
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Vocabulary",
        version="V2.0"
    )
