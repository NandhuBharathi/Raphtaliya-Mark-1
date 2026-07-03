
import torch


class Padder:

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def pad(self, sequences):

        max_len = max(len(seq) for seq in sequences)

        padded = []

        for seq in sequences:
            padded.append(
                seq + [self.pad_token_id] * (max_len - len(seq))
            )

        return torch.tensor(
            padded,
            dtype=torch.long
        )
