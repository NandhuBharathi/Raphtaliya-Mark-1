
import re
import unicodedata


class Normalizer:

    def normalize(self, text):

        text = unicodedata.normalize("NFC", text)

        text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()
