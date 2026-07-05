
import time
import torch
from tqdm.auto import tqdm
from torch.utils.data import (
    DataLoader,
    random_split
)

from raphtaliya.pipeline import DataPipeline
from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.trainer import Trainer
from raphtaliya.evaluator import Evaluator
from raphtaliya.checkpoint import CheckpointManager
from raphtaliya.inference import InferenceEngine
from raphtaliya.dataloader import LanguageDataset


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("🦊 Raphtaliya Mark-1 Training V2")
print("=" * 60)
print(f"Device : {device}")

if device.type == "cuda":

    print(
        "GPU    :",
        torch.cuda.get_device_name(0)
    )

    memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024 ** 3
    )

    print(f"Memory : {memory:.2f} GB")

print("=" * 60)


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

train_size = int(len(dataset) * 0.9)
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False
)

vocab_size = data["tokenizer"].vocab_size()

print(f"Vocabulary         : {vocab_size:,}")
print(f"Total Samples      : {len(dataset):,}")
print(f"Train Samples      : {len(train_dataset):,}")
print(f"Validation Samples : {len(validation_dataset):,}")

print("=" * 60)


# ==========================================
# Model
# ==========================================

model = RaphtaliyaMark1(
    vocab_size=vocab_size,
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=4,
    max_sequence_length=512
)

trainer = Trainer(
    model,
    device=device
)

model = trainer.model

parameters = trainer.model_parameters()

print(
    f"Parameters : {parameters:,} ({parameters/1e6:.2f} M)"
)

print("=" * 60)


# ==========================================
# Training Configuration
# ==========================================

EPOCHS = 10

best_train_loss = float("inf")
best_validation_loss = float("inf")

checkpoint = CheckpointManager()

training_start = time.time()

print("Training Started...\\n")

# -------- END OF PART 1 --------
# ==========================================
# Training
# ==========================================

for epoch in range(EPOCHS):

    epoch_start = time.time()

    total_train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=True
    )

    model.train()

    for inputs, targets in progress:

        loss = trainer.train_step(
            inputs,
            targets
        )

        total_train_loss += loss

        average_loss = total_train_loss / (progress.n + 1)

        progress.set_postfix(
            train_loss=f"{average_loss:.4f}",
            lr=f"{trainer.learning_rate():.2e}"
        )

    train_loss = total_train_loss / len(train_loader)

    # ==========================================
    # Validation
    # ==========================================

    evaluator = Evaluator(
        model,
        device=device
    )

    validation_result = evaluator.evaluate(
        validation_loader
    )

    validation_loss = validation_result["loss"]

    epoch_time = time.time() - epoch_start

    average_epoch_time = (
        time.time() - training_start
    ) / (epoch + 1)

    remaining = average_epoch_time * (
        EPOCHS - epoch - 1
    )

    print("=" * 60)
    print(f"Epoch            : {epoch + 1}/{EPOCHS}")
    print(f"Train Loss       : {train_loss:.4f}")
    print(f"Validation Loss  : {validation_loss:.4f}")
    print(f"Perplexity       : {validation_result['perplexity']:.4f}")

    print(
        f"Epoch Time       : "
        f"{int(epoch_time // 60):02d}:"
        f"{int(epoch_time % 60):02d}"
    )

    print(
        f"ETA              : "
        f"{int(remaining // 60):02d}:"
        f"{int(remaining % 60):02d}"
    )

    # ==========================================
    # Best Checkpoint
    # ==========================================

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_train_loss = train_loss

        checkpoint.save_best(
            model=model,
            optimizer=trainer.optimizer,
            epoch=epoch + 1,
            loss=validation_loss,
            metadata={
                "vocabulary": vocab_size,
                "sequence_length": 64,
                "batch_size": 8,
                "train_loss": train_loss,
                "validation_loss": validation_loss
            }
        )

        print("✅ Best Checkpoint Saved")

    checkpoint.save_latest(
        model=model,
        optimizer=trainer.optimizer,
        epoch=epoch + 1,
        loss=train_loss,
        metadata={
            "validation_loss": validation_loss
        }
    )

    print("=" * 60)

# -------- END OF PART 2 --------
# ==========================================
# Final Checkpoint
# ==========================================

print("\nSaving Final Model...")

checkpoint.save(
    model=model,
    optimizer=trainer.optimizer,
    epoch=EPOCHS,
    loss=best_validation_loss,
    filename="mark1_102books.pt",
    metadata={
        "vocabulary": vocab_size,
        "epochs": EPOCHS,
        "batch_size": 8,
        "sequence_length": 64,
        "train_loss": best_train_loss,
        "validation_loss": best_validation_loss
    }
)

print("Final Checkpoint Saved")


# ==========================================
# Inference Test
# ==========================================

print("\nLoading Best Checkpoint For Inference...")

checkpoint.load_best(
    model=model,
    optimizer=None,
    device=device
)

engine = InferenceEngine(
    model=model,
    tokenizer=data["tokenizer"],
    device=device
)

print("\nInference Test")
print("=" * 60)

prompts = [
    "Hello",
    "Who are you",
    "What is Artificial Intelligence",
    "Once upon a time"
]

for prompt in prompts:

    print(f"\nPrompt : {prompt}")

    response = engine.generate(
        prompt,
        max_new_tokens=30
    )

    print(f"Response : {response}")

print("=" * 60)

# -------- END OF PART 3 --------
# ==========================================
# Training Summary
# ==========================================

total_time = time.time() - training_start

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("\n" + "=" * 60)
print("🦊 Raphtaliya Mark-1 Training Completed")
print("=" * 60)

print(f"Device              : {device}")
print(f"Epochs              : {EPOCHS}")
print(f"Vocabulary          : {vocab_size:,}")
print(f"Training Steps      : {trainer.training_steps}")
print(f"Best Train Loss     : {best_train_loss:.4f}")
print(f"Best Validation Loss: {best_validation_loss:.4f}")
print(f"Validation Perplexity : {validation_result['perplexity']:.4f}")
print(f"Parameters          : {trainer.model_parameters():,}")

print(
    f"Training Time       : "
    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
)

print("=" * 60)
print("Training Finished Successfully")
print("=" * 60)

print("\nNext Recommended Steps")
print("- Continue training from checkpoints/best.pt")
print("- Add high-quality dialogue datasets")
print("- Add QA datasets")
print("- Add programming datasets")
print("- Improve reasoning")
print("- Build Raphtaliya Core")

# ==========================================
# Resume Training Information
# ==========================================

print("\nResume Training")

if checkpoint.exists("best.pt"):

    info = checkpoint.checkpoint_info("best.pt")

    print("=" * 60)
    print("Best Checkpoint")
    print("=" * 60)
    print(f"Epoch      : {info['epoch']}")
    print(f"Loss       : {info['loss']:.4f}")
    print(f"Created At : {info['created_at']}")
    print("=" * 60)

else:

    print("No best checkpoint found.")

print("\n🦊 Raphtaliya Mark-1 V2 Ready")

# -------- END OF PART 4 --------
