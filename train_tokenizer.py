
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from pathlib import Path

# ==========================================
# Create Tokenizer
# ==========================================

tokenizer = Tokenizer(
    BPE(
        unk_token="<UNK>"
    )
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=True
)

# ==========================================
# Trainer
# ==========================================

trainer = BpeTrainer(
    vocab_size=32768,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ],
    show_progress=True
)

# ==========================================
# Dataset
# ==========================================

files = [
    str(file)
    for file in Path("dataset/train").rglob("*.txt")
]

print("=" * 60)
print("Raphtaliya BPE Trainer")
print("=" * 60)
print(f"Books : {len(files)}")
print("=" * 60)

# ==========================================
# Train
# ==========================================

tokenizer.train(files, trainer)

# ==========================================
# Save
# ==========================================

Path("tokenizer").mkdir(exist_ok=True)

tokenizer.save("tokenizer/tokenizer.json")

print("=" * 60)
print("Training Completed")
print(f"Vocabulary Size : {tokenizer.get_vocab_size():,}")
print("Saved : tokenizer/tokenizer.json")
print("=" * 60)
