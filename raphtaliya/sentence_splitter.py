
import re


class SentenceSplitter:

    def split(self, text):

        # Split into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        cleaned = []

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 3:
                continue

            cleaned.append(sentence)

        return cleaned
