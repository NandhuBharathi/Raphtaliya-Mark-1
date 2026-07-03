
from raphtaliya.dataset import Dataset
from raphtaliya.vocabulary import Vocabulary
from raphtaliya.tokenizer import Tokenizer
from raphtaliya.sequence import SequenceBuilder
from raphtaliya.padding import Padder


class DataPipeline:

    def __init__(self, dataset_path, sequence_length=32):

        self.dataset_path = dataset_path
        self.sequence_length = sequence_length

    def build(self):

        dataset = Dataset(self.dataset_path)

        texts = dataset.load()

        vocabulary = Vocabulary()
        vocabulary.build(texts)

        tokenizer = Tokenizer(vocabulary)

        builder = SequenceBuilder(
            tokenizer,
            self.sequence_length
        )

        inputs, targets = builder.build(texts)

        padder = Padder()

        inputs = padder.pad(inputs)
        targets = padder.pad(targets)

        return {
            "texts": texts,
            "vocabulary": vocabulary,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "targets": targets
        }
