
import torch


class InferenceEngine:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

        self.model.eval()

    @torch.no_grad()
    def predict_next(self, text):

        token_ids = self.tokenizer.encode(text)

        inputs = torch.tensor(
            [token_ids],
            dtype=torch.long
        )

        logits = self.model(inputs)

        next_token = torch.argmax(
            logits[0, -1]
        ).item()

        return self.tokenizer.vocabulary.get_word(
            next_token
        )
