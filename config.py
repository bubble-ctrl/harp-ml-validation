"""
config.py — Shared configuration for HARP‑ML validation.
"""

import torch

# Model configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
USE_4BIT = True                      # Enable 4-bit quantization (reduces memory to ~1-2GB)
FALLBACK_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "auto"                  # requires accelerate; distributes across GPU/CPU

# Evaluation settings
NUM_PBT_RUNS = 100
GNC_NUM_STEPS = 40
RANDOM_SEED = 42

# Paths (relative to project root)
DATASET_PATH = "data/manual_test_dataset.py"