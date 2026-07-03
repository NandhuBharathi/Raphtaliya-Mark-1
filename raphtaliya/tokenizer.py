
from raphtaliya.normalizer import Normalizer
from raphtaliya.pretokenizer import PreTokenizer
from raphtaliya.vocabulary import Vocabulary


class Tokenizer:

    def __init__(self, vocabulary):

        self.normalizer = Normalizer()
        self.pretokenizer = PreTokenizer()
        self.vocabulary = vocabulary

    def encode(self, text):

        text = self.normalizer.normalize(text)

        tokens = self.pretokenizer.tokenize(text)

        ids = [
            self.vocabulary.get_id(token)
            for token in tokens
        ]

        return ids

    def decode(self, ids):

        tokens = [
            self.vocabulary.get_word(token_id)
            for token_id in ids
        ]

        return " ".join(tokens)
