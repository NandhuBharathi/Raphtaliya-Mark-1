
from raphtaliya.dataset import Dataset
from raphtaliya.document_splitter import DocumentSplitter
from raphtaliya.cleaner import TextCleaner
from raphtaliya.sentence_splitter import SentenceSplitter
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

        books = dataset.load()

        document_splitter = DocumentSplitter()
        cleaner = TextCleaner()
        sentence_splitter = SentenceSplitter()

        texts = []

        for book in books:

            story = document_splitter.split(book)

            story = cleaner.clean(story)

            sentences = sentence_splitter.split(story)

            texts.extend(sentences)

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
