import time
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("🦊 Raphtaliya Mark-1 Training V2")
print("=" * 60)
print(f"Device : {device}")
if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))
    memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"Memory : {memory:.2f} GB")
print("=" * 60)

# ==========================================
# Dataset
# ==========================================
pipeline = DataPipeline(dataset_path="dataset/train", sequence_length=64)
data = pipeline.build()
dataset = LanguageDataset(data["inputs"].tolist(), data["targets"].tolist())
loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"Vocabulary : {data['vocabulary'].size():,}")
print(f"Samples    : {len(dataset):,}")
print("=" * 60)

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
trainer = Trainer(model, device=device)
model = trainer.model
parameters = trainer.model_parameters()
print(f"Parameters : {parameters:,} ({parameters/1e6:.2f} M)")
print("=" * 60)

# ==========================================
# Training
# ==========================================
EPOCHS = 10
best_loss = float("inf")
checkpoint = CheckpointManager()
training_start = time.time()
print("Training Started...\n")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
    for inputs, targets in progress:
        loss = trainer.train_step(inputs, targets)
        total_loss += loss
        average_loss = total_loss / (progress.n + 1)
        progress.set_postfix(
            loss=f"{average_loss:.4f}",
            lr=f"{trainer.learning_rate():.2e}"
        )
    average_epoch_time = (time.time() - training_start) / (epoch + 1)
    remaining = average_epoch_time * (EPOCHS - epoch - 1)
    epoch_loss = total_loss / len(loader)
    epoch_time = time.time() - epoch_start
    
    print("=" * 60)
    print(f"Epoch      : {epoch + 1}/{EPOCHS}")
    print(f"Loss       : {epoch_loss:.4f}")
    epoch_minutes = int(epoch_time // 60)
    epoch_seconds = int(epoch_time % 60)

    eta_minutes = int(remaining // 60)
    eta_seconds = int(remaining % 60)

    print(f"Epoch Time : {epoch_minutes:02d}:{epoch_seconds:02d}")
    print(f"ETA : {eta_minutes:02d}:{eta_seconds:02d}")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        checkpoint.save_best(
            model=model,
            optimizer=trainer.optimizer,
            epoch=epoch + 1,
            loss=epoch_loss,
            metadata={
                "vocabulary": data["vocabulary"].size(),
                "sequence_length": 64,
                "batch_size": 8
            }
        )
        print("Best Checkpoint Saved")
    checkpoint.save_latest(model=model, optimizer=trainer.optimizer, epoch=epoch + 1, loss=epoch_loss)
    print("=" * 60)

# ==========================================
# Evaluation
# ==========================================
print("\nRunning Evaluation...")
evaluator = Evaluator(model, device=device)
result = evaluator.evaluate(loader)
print("=" * 60)
print("Evaluation")
print("=" * 60)
print(f"Loss        : {result['loss']}")
print(f"Perplexity  : {result['perplexity']}")
print(f"Batches     : {result['batches']}")
print(f"Device      : {result['device']}")
print("=" * 60)

# ==========================================
# Final Checkpoint
# ==========================================
checkpoint.save(
    model=model,
    optimizer=trainer.optimizer,
    epoch=EPOCHS,
    loss=result["loss"],
    filename="mark1_102books.pt",
    metadata={
        "vocabulary": data["vocabulary"].size(),
        "epochs": EPOCHS,
        "batch_size": 8,
        "sequence_length": 64
    }
)
print("Final Checkpoint Saved")

# ==========================================
# Inference Test
# ==========================================
engine = InferenceEngine(model, data["tokenizer"], device=device)
print("\nInference Test")
print("=" * 60)
prompts = ["Hello", "Who are you", "Once upon a time"]
for prompt in prompts:
    print(f"\nPrompt : {prompt}")
    response = engine.generate(prompt, max_new_tokens=30)
    print(f"Response : {response}")

# ==========================================
# Training Summary
# ==========================================
total_time = time.time() - training_start
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print("\n" + "=" * 60)
print("Raphtaliya Mark-1 Training Completed")
print("=" * 60)
print(f"Device           : {device}")
print(f"Epochs           : {EPOCHS}")
print(f"Vocabulary       : {data['vocabulary'].size():,}")
print(f"Training Steps   : {trainer.training_steps}")
print(f"Best Loss        : {best_loss:.4f}")
print(f"Final Loss       : {result['loss']}")
print(f"Perplexity       : {result['perplexity']}")
print(f"Parameters       : {trainer.model_parameters():,}")
print(f"Training Time    : {hours:02d}:{minutes:02d}:{seconds:02d}")
print("=" * 60)
print("Training Finished Successfully")
print("=" * 60)
