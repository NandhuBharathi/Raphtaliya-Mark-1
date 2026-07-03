
class DatasetStatistics:

    def analyze(self, texts, vocabulary):

        total_sentences = len(texts)

        total_words = sum(
            len(sentence.split())
            for sentence in texts
        )

        average_sentence_length = (
            total_words / total_sentences
            if total_sentences > 0
            else 0
        )

        longest_sentence = max(
            texts,
            key=len,
            default=""
        )

        return {
            "total_sentences": total_sentences,
            "total_words": total_words,
            "vocabulary_size": vocabulary.size(),
            "average_sentence_length": round(
                average_sentence_length,
                2
            ),
            "longest_sentence_length": len(
                longest_sentence.split()
            )
        }

    def print(self, stats):

        print("=" * 40)
        print("DATASET STATISTICS")
        print("=" * 40)

        print(f"Total Sentences        : {stats['total_sentences']}")
        print(f"Total Words            : {stats['total_words']}")
        print(f"Vocabulary Size        : {stats['vocabulary_size']}")
        print(f"Average Sentence Words : {stats['average_sentence_length']}")
        print(f"Longest Sentence Words : {stats['longest_sentence_length']}")

        print("=" * 40)
