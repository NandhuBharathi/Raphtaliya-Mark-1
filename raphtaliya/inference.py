import torch


class InferenceEngine:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        self.special_tokens = {
            "<PAD>",
            "<BOS>",
            "<UNK>"
        }

    @torch.no_grad()
    def generate(
        self,
        text,
        max_new_tokens=20,
        temperature=1.0,
        top_k=10
    ):

        token_ids = self.tokenizer.encode(text)

        generated = 0

        for _ in range(max_new_tokens):

            inputs = torch.tensor(
                [token_ids],
                dtype=torch.long
            )

            logits = self.model(inputs)

            logits = logits[0, -1] / temperature

            values, indices = torch.topk(
                logits,
                k=min(top_k, logits.size(-1))
            )

            probs = torch.softmax(values, dim=-1)

            next_token = None

            while next_token is None:

                sampled = torch.multinomial(
                    probs,
                    num_samples=1
                ).item()

                candidate = indices[sampled].item()

                word = self.tokenizer.vocabulary.get_word(candidate)

                if word == "<EOS>":
                    return self.tokenizer.decode(token_ids)

                if word in self.special_tokens:
                    continue

                next_token = candidate

            token_ids.append(next_token)
            generated += 1

        return self.tokenizer.decode(token_ids)

    def predict_next_token(
        self,
        text,
        temperature=1.0,
        top_k=10
    ):

        token_ids = self.tokenizer.encode(text)

        inputs = torch.tensor(
            [token_ids],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = self.model(inputs)

            logits = logits[0, -1] / temperature

            values, indices = torch.topk(
                logits,
                k=min(top_k, logits.size(-1))
            )

            probs = torch.softmax(values, dim=-1)

        predictions = []

        for probability, index in zip(probs.tolist(), indices.tolist()):

            predictions.append({
                "token": self.tokenizer.vocabulary.get_word(index),
                "probability": round(probability, 4)
            })

        return predictions

    def model_info(self):

        return {
            "model": self.model.__class__.__name__,
            "vocabulary_size": self.tokenizer.vocab_size()
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Inference",
        version="V2.0"
    )
