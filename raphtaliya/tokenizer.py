from raphtaliya.normalizer import Normalizer
from raphtaliya.pretokenizer import PreTokenizer
from raphtaliya.vocabulary import Vocabulary


class Tokenizer:

    def __init__(self, vocabulary):

        self.normalizer = Normalizer()
        self.pretokenizer = PreTokenizer()
        self.vocabulary = vocabulary

    def tokenize(self, text):

        text = self.normalizer.normalize(text)

        return self.pretokenizer.tokenize(text)

    def detokenize(self, tokens):

        return " ".join(tokens)

    def encode(self, text):

        tokens = self.tokenize(text)

        return [
            self.vocabulary.get_id(token)
            for token in tokens
        ]

    def encode_batch(self, texts):

        return [
            self.encode(text)
            for text in texts
        ]

    def decode(self, ids):

        tokens = [
            self.vocabulary.get_word(token_id)
            for token_id in ids
        ]

        return self.detokenize(tokens)

    def decode_batch(self, batch_ids):

        return [
            self.decode(ids)
            for ids in batch_ids
        ]

    def contains(self, token):

        return token in self.vocabulary.word_to_id

    def vocab_size(self):

        return self.vocabulary.size()

    def statistics(self):

        return {
            "vocabulary_size": self.vocabulary.size(),
            "special_tokens": self.vocabulary.special_tokens
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Tokenizer",
        version="V2.0"
    )
