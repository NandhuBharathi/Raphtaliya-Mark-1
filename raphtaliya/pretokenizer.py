
import string


class PreTokenizer:

    def __init__(self):

        self.punctuation = set(string.punctuation)

    def tokenize(self, text):

        tokens = []
        current = ""

        for ch in text:

            if ch.isspace():

                if current:
                    tokens.append(current)
                    current = ""

            elif ch in self.punctuation:

                if current:
                    tokens.append(current)
                    current = ""

                tokens.append(ch)

            else:

                current += ch

        if current:
            tokens.append(current)

        return tokens
