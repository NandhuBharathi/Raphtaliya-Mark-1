
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
            logits[0, -1],
            dim=-1
        ).item()

        return next_token

    @torch.no_grad()
    def generate(self, text, max_new_tokens=20):

        token_ids = self.tokenizer.encode(text)

        for _ in range(max_new_tokens):

            inputs = torch.tensor(
                [token_ids],
                dtype=torch.long
            )

            logits = self.model(inputs)

            next_token = torch.argmax(
                logits[0, -1],
                dim=-1
            ).item()

            token_ids.append(next_token)

        return self.tokenizer.decode(token_ids)
