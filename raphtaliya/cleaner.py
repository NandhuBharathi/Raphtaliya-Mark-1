
import re


class TextCleaner:

    def clean(self, text):

        # Remove Project Gutenberg header
        start = text.find("*** START OF")
        if start != -1:
            start = text.find("\n", start)
            text = text[start:]

        # Remove Project Gutenberg footer
        end = text.find("*** END OF")
        if end != -1:
            text = text[:end]

        # Normalize line endings
        text = text.replace("\r", "")

        # Remove extra blank lines
        text = re.sub(r"\n\s*\n", "\n", text)

        # Remove extra spaces
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()
