
from torch.utils.data import DataLoader

from raphtaliya.pipeline import DataPipeline
from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.trainer import Trainer
from raphtaliya.dataloader import LanguageDataset
from raphtaliya.evaluator import Evaluator
from raphtaliya.checkpoint import CheckpointManager
from raphtaliya.inference import InferenceEngine


# ==========================================
# Dataset
# ==========================================

pipeline = DataPipeline(
    dataset_path="dataset/train",
    sequence_length=64
)

data = pipeline.build()


dataset = LanguageDataset(
    data["inputs"].tolist(),
    data["targets"].tolist()
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# ==========================================
# Model
# ==========================================

model = RaphtaliyaMark1(
    vocab_size=data["vocabulary"].size(),
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=4,
    max_sequence_length=512
)


# ==========================================
# Trainer
# ==========================================

trainer = Trainer(model)

EPOCHS = 10

print("=" * 60)
print("Raphtaliya Mark-1 Training")
print("=" * 60)

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
        f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {average_loss:.4f}"
    )


# ==========================================
# Evaluation
# ==========================================

print("\nRunning Evaluation...")

evaluator = Evaluator(model)

result = evaluator.evaluate(loader)

print(result)


# ==========================================
# Save Checkpoint
# ==========================================

checkpoint = CheckpointManager()

checkpoint.save(
    model,
    filename="mark1_102books.pt"
)

print("Checkpoint Saved.")


# ==========================================
# Inference Test
# ==========================================

engine = InferenceEngine(
    model,
    data["tokenizer"]
)

print("\nInference Test")
print("-" * 60)

print(
    engine.generate(
        "Hello",
        max_new_tokens=30
    )
)

print("=" * 60)
print("Training Completed Successfully")
print("=" * 60)
