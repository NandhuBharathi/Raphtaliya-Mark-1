
from collections import Counter

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
