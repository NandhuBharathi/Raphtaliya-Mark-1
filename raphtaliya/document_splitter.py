
import re


class DocumentSplitter:

    def split(self, text):

        # Remove Gutenberg header
        start = re.search(
            r"\*\*\* START OF.*?\*\*\*",
            text,
            flags=re.DOTALL
        )

        if start:
            text = text[start.end():]

        # Remove Gutenberg footer
        end = re.search(
            r"\*\*\* END OF.*",
            text,
            flags=re.DOTALL
        )

        if end:
            text = text[:end.start()]

        # Remove illustration tags
        text = re.sub(r"\[.*?\]", "", text)

        # Find all Chapter I occurrences
        chapters = list(
            re.finditer(
                r"CHAPTER\s+I\b",
                text,
                flags=re.IGNORECASE
            )
        )

        # Skip Contents page
        if len(chapters) >= 2:
            text = text[chapters[1].start():]

        # Normalize spaces
        text = text.replace("\r", "")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        return text.strip()
