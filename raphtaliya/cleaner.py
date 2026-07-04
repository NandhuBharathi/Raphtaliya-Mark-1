
import re


class TextCleaner:

    def clean(self, text):

        # -----------------------------
        # Remove Project Gutenberg Header
        # -----------------------------
        start_marker = "*** START OF"

        start = text.find(start_marker)

        if start != -1:
            start = text.find("\n", start)
            if start != -1:
                text = text[start + 1:]


        # -----------------------------
        # Remove Project Gutenberg Footer
        # -----------------------------
        end_markers = [
            "*** END OF",
            "End of the Project Gutenberg",
            "End of Project Gutenberg"
        ]

        for marker in end_markers:

            end = text.find(marker)

            if end != -1:
                text = text[:end]
                break


        # -----------------------------
        # Remove Illustration Tags
        # -----------------------------
        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )


        # -----------------------------
        # Remove Contents Page
        # -----------------------------
        text = re.sub(
            r"Contents.*?(CHAPTER|ACT I|BOOK I|PART I)",
            r"\1",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )


        # -----------------------------
        # Remove Multiple Blank Lines
        # -----------------------------
        text = text.replace("\r", "")

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )


        # -----------------------------
        # Normalize Spaces
        # -----------------------------
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )


        return text.strip()


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Cleaner",
        version="V2.0"
    )
