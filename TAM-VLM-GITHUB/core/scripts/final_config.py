"""Single source of truth for TAM-VLM experiments.

All paths can be overridden using environment variables.

Example:

export TAMVLM_DATA_ROOT=/path/to/dataset
export TAMVLM_EMB_DIR=/path/to/embeddings
export TAMVLM_OUT_DIR=./results
export TAMVLM_CLIP_WEIGHTS=/path/to/open_clip_weights
"""

from pathlib import Path
import os


# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent


# Dataset location
DATA_ROOT = Path(
    os.environ.get(
        "TAMVLM_DATA_ROOT",
        PROJECT_DIR / "data"
    )
)


# Precomputed embeddings
EMB_DIR = Path(
    os.environ.get(
        "TAMVLM_EMB_DIR",
        DATA_ROOT / "embeddings"
    )
)


# Manifest file
MANIFEST = Path(
    os.environ.get(
        "TAMVLM_MANIFEST",
        DATA_ROOT / "manifest.csv"
    )
)


# Output directory
OUT_DIR = Path(
    os.environ.get(
        "TAMVLM_OUT_DIR",
        PROJECT_DIR / "results"
    )
)


CKPT_DIR = OUT_DIR / "checkpoints"

CACHE_DIR = OUT_DIR / "cache"


# CLIP configuration
CLIP_MODEL = "ViT-B-16"

CLIP_WEIGHTS = os.environ.get(
    "TAMVLM_CLIP_WEIGHTS",
    str(PROJECT_DIR / "checkpoints/open_clip_model.safetensors")
)


# Trigger categories
ALL_TRIGGERS = [
    "traffic cone",
    "red balloon",
    "football",
    "rose",
    "fire hydrant",
    "traffic barrier",
    "bollard",
    "ladder",
    "potted plant",
    "roadside litter",
    "teddy bear",
    "umbrella",
]


# Experimental settings

SEEDS_MAIN = (0, 1, 2, 3, 4)

SEEDS_LOTO = (0, 1, 2)

BATCH_SIZE = 256

MAX_EPOCHS = 60

PATIENCE = 12

LR = 1e-3

WEIGHT_DECAY = 1e-4


# Diversity experiment

DIVERSITY_K = (2, 4, 6, 8, 11)

DIVERSITY_REPS = 3

DIVERSITY_TRAIN_POS_BUDGET = 5000

DIVERSITY_VAL_POS_BUDGET = 1000


# Bootstrap

BOOTSTRAP_REPS = 2000


# Create outputs

for path in (OUT_DIR, CKPT_DIR, CACHE_DIR):
    path.mkdir(
        parents=True,
        exist_ok=True
    )
