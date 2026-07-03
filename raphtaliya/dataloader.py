
import torch
from torch.utils.data import Dataset


class LanguageDataset(Dataset):

    def __init__(self, inputs, targets):

        self.inputs = inputs
        self.targets = targets

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        return (
            torch.tensor(
                self.inputs[index],
                dtype=torch.long
            ),
            torch.tensor(
                self.targets[index],
                dtype=torch.long
            )
        )
