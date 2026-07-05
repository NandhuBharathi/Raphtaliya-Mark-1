import torch

from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.tokenizer import RaphtaliyaTokenizer
from raphtaliya.inference import InferenceEngine
from raphtaliya.checkpoint import CheckpointManager


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("🦊 Raphtaliya Mark-1 Inference")
print("=" * 60)
print("Device :", device)
print("=" * 60)


# ==========================================
# Tokenizer
# ==========================================

tokenizer = RaphtaliyaTokenizer(
    "tokenizer/tokenizer.json"
)

vocab_size = tokenizer.vocab_size()


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


# ==========================================
# Load Checkpoint
# ==========================================

checkpoint = CheckpointManager()

info = checkpoint.load(
    model=model,
    filename="best.pt",
    device=device
)

print("Checkpoint Loaded")
print("Epoch :", info["epoch"])
print("Loss  :", info["loss"])
print("=" * 60)


# ==========================================
# Inference Engine
# ==========================================

engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    device=device
)


# ==========================================
# Chat Loop
# ==========================================

print("Type 'exit' to quit.\n")

while True:

    prompt = input("You : ")

    if prompt.lower() == "exit":
        break

    response = engine.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=20
    )

    print(f"\nRaphtaliya : {response}\n")