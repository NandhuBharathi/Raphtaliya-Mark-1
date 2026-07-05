
# ==========================================
# Raphtaliya Mark-1 Configuration
# ==========================================

# Version

VERSION = "Mark-1 V2"


# ==========================================
# Training
# ==========================================

EPOCHS = 10

BATCH_SIZE = 8

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

GRADIENT_CLIP = 1.0

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# ==========================================
# Model
# ==========================================

EMBEDDING_DIM = 256

NUM_HEADS = 8

HIDDEN_DIM = 1024

NUM_LAYERS = 4

MAX_SEQUENCE_LENGTH = 512


# ==========================================
# Dataset
# ==========================================

SEQUENCE_LENGTH = 64

DATASET_PATH = "dataset/train"


# ==========================================
# Checkpoints
# ==========================================

CHECKPOINT_DIRECTORY = "checkpoints"

BEST_CHECKPOINT = "best.pt"

LATEST_CHECKPOINT = "latest.pt"

FINAL_CHECKPOINT = "mark1_102books.pt"


# ==========================================
# Tokenizer
# ==========================================

TOKENIZER_PATH = "tokenizer/tokenizer.json"

TOKENIZER_PATH = "tokenizer"


# ==========================================
# Device
# ==========================================

DEFAULT_DEVICE = "cuda"


# ==========================================
# Inference
# ==========================================

MAX_NEW_TOKENS = 30

TEMPERATURE = 1.0

TOP_K = 50

TOP_P = 0.95


# ==========================================
# Logging
# ==========================================

PRINT_PROGRESS = True

SAVE_EVERY_EPOCH = True


# ==========================================
# Metadata
# ==========================================

AUTHOR = "NANDHAKUMAR M.S"

MODEL_NAME = "Raphtaliya Mark-1"
