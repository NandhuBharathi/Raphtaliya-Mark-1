import os
import torch

from huggingface_hub import hf_hub_download

from raphtaliya.model import RaphtaliyaMark1
from raphtaliya.tokenizer import RaphtaliyaTokenizer
from raphtaliya.inference import InferenceEngine


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("🦊 Raphtaliya Mark-1 Inference")
print("=" * 60)
print("Device :", device)
print("=" * 60)


# ============================================================
# Download / Load Checkpoint
# ============================================================

CHECKPOINT_DIR = "checkpoints"
TOKENIZER_DIR = "tokenizer"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)


checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best.pt"
)

if not os.path.exists(checkpoint_path):

    print("Downloading model from Hugging Face...")

    checkpoint_path = hf_hub_download(
        repo_id="nandhakumarms/Raphtaliya-Mark-1",
        filename="best.pt",
        local_dir=CHECKPOINT_DIR
    )

    print("Download Completed.")

else:

    print("Using Local Checkpoint.")


tokenizer_path = os.path.join(
    TOKENIZER_DIR,
    "tokenizer.json"
)

if not os.path.exists(tokenizer_path):

    print("Downloading Tokenizer...")

    tokenizer_path = hf_hub_download(
        repo_id="nandhakumarms/Raphtaliya-Mark-1",
        filename="tokenizer/tokenizer.json",
        local_dir=TOKENIZER_DIR
    )

    print("Tokenizer Downloaded.")

else:

    print("Using Local Tokenizer.")
# ============================================================
# Load Tokenizer
# ============================================================

tokenizer = RaphtaliyaTokenizer(
    tokenizer_path=tokenizer_path
)

vocab_size = tokenizer.vocab_size()


# ============================================================
# Create Model
# ============================================================

model = RaphtaliyaMark1(
    vocab_size=vocab_size,
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=4,
    max_sequence_length=512
)


# ============================================================
# Load Checkpoint
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("\nCheckpoint Loaded Successfully")
print(f"Epoch : {checkpoint['epoch']}")
print(f"Loss  : {checkpoint['loss']:.4f}")
print("=" * 60)


# ============================================================
# Create Inference Engine
# ============================================================

engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    device=device
)


# ============================================================
# Interactive Chat
# ============================================================

print("🦊 Raphtaliya is Ready!")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    prompt = input("\nYou : ")

    if prompt.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    response = engine.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=20
    )

    print(f"\nRaphtaliya : {response}")
