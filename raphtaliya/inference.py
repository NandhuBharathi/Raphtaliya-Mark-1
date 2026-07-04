
import torch


class InferenceEngine:

    def __init__(
        self,
        model,
        tokenizer,
        device=None
    ):

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.model.eval()

        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.unk_id = self.tokenizer.token_to_id("<UNK>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    @torch.no_grad()
    def generate(
        self,
        text,
        max_new_tokens=20,
        temperature=1.0,
        top_k=10
    ):

        token_ids = self.tokenizer.encode(text)

        for _ in range(max_new_tokens):

            inputs = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=self.device
            )

            logits = self.model(inputs)

            logits = logits[0, -1] / temperature

            values, indices = torch.topk(
                logits,
                k=min(top_k, logits.size(-1))
            )

            probs = torch.softmax(
                values,
                dim=-1
            )

            next_token = None

            while next_token is None:

                sampled = torch.multinomial(
                    probs,
                    1
                ).item()

                candidate = indices[
                    sampled
                ].item()

                if candidate == self.eos_id:
                    return self.tokenizer.decode(
                        token_ids
                    )

                if candidate in (
                    self.pad_id,
                    self.unk_id,
                    self.bos_id
                ):
                    continue

                next_token = candidate

            token_ids.append(next_token)

        return self.tokenizer.decode(
            token_ids
        )

    @torch.no_grad()
    def predict_next_token(
        self,
        text,
        temperature=1.0,
        top_k=10
    ):

        token_ids = self.tokenizer.encode(text)

        inputs = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device
        )

        logits = self.model(inputs)

        logits = logits[0, -1] / temperature

        values, indices = torch.topk(
            logits,
            k=min(top_k, logits.size(-1))
        )

        probs = torch.softmax(
            values,
            dim=-1
        )

        predictions = []

        for probability, index in zip(
            probs.tolist(),
            indices.tolist()
        ):

            predictions.append(
                {
                    "token": self.tokenizer.id_to_token(index),
                    "probability": round(
                        probability,
                        4
                    )
                }
            )

        return predictions

    def model_info(self):

        return {

            "model":
                self.model.__class__.__name__,

            "device":
                str(self.device),

            "vocabulary_size":
                self.tokenizer.vocab_size(),

            "parameters":
                sum(
                    p.numel()
                    for p in self.model.parameters()
                )
        }


if __name__ == "__main__":

    from raphtaliya.utils import show_upgrade

    show_upgrade(
        module="Inference",
        version="V4.0"
    )
