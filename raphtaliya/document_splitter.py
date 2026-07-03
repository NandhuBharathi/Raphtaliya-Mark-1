
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

        # Remove common front matter
        patterns = [
            r"The Project Gutenberg eBook.*?\n",
            r"This eBook.*?\n",
            r"by .*?\n",
            r"THE MILLENNIUM FULCRUM EDITION.*?\n",
            r"Contents.*?(?=CHAPTER|BOOK|PART)",
        ]

        for pattern in patterns:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE
            )

        # Remove illustration tags
        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )

        # Normalize spaces
        text = text.replace("\r", "")

        text = re.sub(
            r"\n\s*\n",
            "\n",
            text
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        return text.strip()
