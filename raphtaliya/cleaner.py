
import re


class TextCleaner:

    def clean(self, text):

        # Remove Gutenberg header
        start_patterns = [
            "*** START OF",
            "CHAPTER I.",
            "CHAPTER I\n",
            "CHAPTER I\r\n"
        ]

        start_index = 0

        for pattern in start_patterns:

            index = text.find(pattern)

            if index != -1:
                start_index = index
                break

        text = text[start_index:]


        # Remove Gutenberg footer
        end_patterns = [
            "*** END OF",
            "End of the Project Gutenberg",
            "End of Project Gutenberg"
        ]

        end_index = len(text)

        for pattern in end_patterns:

            index = text.find(pattern)

            if index != -1:
                end_index = index
                break

        text = text[:end_index]


        # Remove illustration tags
        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )


        # Remove Contents page
        text = re.sub(
            r"Contents.*?CHAPTER I",
            "CHAPTER I",
            text,
            flags=re.DOTALL
        )


        # Remove chapter headings
        text = re.sub(
            r"CHAPTER\s+[IVXLC]+\.*",
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
