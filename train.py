
from raphtaliya.pipeline import DataPipeline
from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.trainer import Trainer
from raphtaliya.dataloader import LanguageDataset

from torch.utils.data import DataLoader


pipeline = DataPipeline(
    dataset_path="dataset/train",
    sequence_length=8
)

data = pipeline.build()


dataset = LanguageDataset(
    data["inputs"].tolist(),
    data["targets"].tolist()
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


model = RaphtaliyaMark1(
    vocab_size=data["vocabulary"].size(),
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=4,
    max_sequence_length=512
)

trainer = Trainer(model)


EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0.0

    for inputs, targets in loader:

        loss = trainer.train_step(
            inputs,
            targets
        )

        total_loss += loss

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1} | Loss: {average_loss:.4f}"
    )
