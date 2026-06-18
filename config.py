from pathlib import Path
import torch

# =====================================================
# Project
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent

# =====================================================
# Audio / Feature Parameters
# =====================================================

SAMPLE_RATE = 32000
CLIP_SAMPLES = SAMPLE_RATE * 4

WINDOW_SIZE = 1024
HOP_SIZE = 320

MEL_BINS = 64
FMIN = 50
FMAX = 14000

NUM_CLASSES = 10

# =====================================================
# Domain Selection Thresholds
# =====================================================

D2_THRESHOLD = 473.2581
D3_THRESHOLD = 363.4958

# =====================================================
# Dataset Paths
# =====================================================

DATA_DIR = ROOT_DIR / "data" / "task7_data"

AUDIO_DIR = DATA_DIR

TEST_CSV = (
    DATA_DIR
    / "evaluation_setup"
    / "development_test.txt"
)

# =====================================================
# Model Paths
# =====================================================

WEIGHTS_DIR = ROOT_DIR / "weights"

TASK1_MODEL_PATH = (
    WEIGHTS_DIR
    / "Takami_OU_task7_1_D2_dictionary.pth"
)

TASK2_MODEL_PATH = (
    WEIGHTS_DIR
    / "Takami_OU_task7_1_D3_dictionary.pth"
)

# =====================================================
# Evaluation
# =====================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "alarm",
    "baby_cry",
    "bark",
    "engine",
    "fire",
    "footsteps",
    "knock",
    "telephone_ringing",
    "piano",
    "speech",
]