
from pathlib import Path
from tokenizers import Tokenizer


class RaphtaliyaTokenizer:

    def __init__(
        self,
        tokenizer_path="tokenizer/tokenizer.json"
    ):

        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():

            raise FileNotFoundError(
                f"Tokenizer not found: {self.tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

    # ==========================================
    # Encode
    # ==========================================

    def encode(self, text):

        return self.tokenizer.encode(text).ids

    # ==========================================
    # Decode
    # ==========================================

    def decode(self, token_ids):

        return self.tokenizer.decode(token_ids)

    # ==========================================
    # Encode Batch
    # ==========================================

    def encode_batch(self, texts):

        outputs = self.tokenizer.encode_batch(texts)

        return [
            output.ids
            for output in outputs
        ]

    # ==========================================
    # Decode Batch
    # ==========================================

    def decode_batch(self, batches):

        return [
            self.decode(ids)
            for ids in batches
        ]

    # ==========================================
    # Vocabulary Size
    # ==========================================

    def vocab_size(self):

        return self.tokenizer.get_vocab_size()

    # ==========================================
    # Token → ID
    # ==========================================

    def token_to_id(self, token):

        return self.tokenizer.token_to_id(token)

    # ==========================================
    # ID → Token
    # ==========================================

    def id_to_token(self, token_id):

        return self.tokenizer.id_to_token(token_id)


if __name__ == "__main__":

    tokenizer = RaphtaliyaTokenizer()

    text = "Raphtaliya is learning AI."

    ids = tokenizer.encode(text)

    print("=" * 60)
    print("Raphtaliya Tokenizer")
    print("=" * 60)
    print("Text :", text)
    print("IDs  :", ids)
    print("Text :", tokenizer.decode(ids))
    print("Vocabulary :", tokenizer.vocab_size())
    print("=" * 60)
