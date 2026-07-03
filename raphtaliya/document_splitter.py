
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
        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )

        # Skip table of contents
        chapters = list(
            re.finditer(
                r"CHAPTER\s+I\b",
                text,
                flags=re.IGNORECASE
            )
        )

        if len(chapters) >= 2:
            text = text[chapters[1].start():]

        # Remove chapter headings
        text = re.sub(
            r"CHAPTER\s+[IVXLC]+\.*\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Normalize line endings
        text = text.replace("\r", "")

        # Remove multiple spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Remove multiple blank lines
        text = re.sub(
            r"\n\s*\n",
            "\n",
            text
        )

        return text.strip()
