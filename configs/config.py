"""
Project Configuration
AI-Business-Risk-Analysis-System
"""

from pathlib import Path

# ==========================
# Project Information
# ==========================

PROJECT_NAME = "AI-Business-Risk-Analysis-System"
VERSION = "2.0.0"

AUTHOR = "Mohamed Shakir"

# ==========================
# Model Configuration
# ==========================

MODEL_NAME = "xlm-roberta-base"

MAX_LENGTH = 128

NUM_SENTIMENT_CLASSES = 3

NUM_ASPECT_CLASSES = 3

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 16

NUM_EPOCHS = 15

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

GRADIENT_CLIP = 1.0

# ==========================
# Dataset Configuration
# ==========================

TRAIN_RATIO = 0.8

VALID_RATIO = 0.1

TEST_RATIO = 0.1

RANDOM_SEED = 42

# ==========================
# Early Stopping
# ==========================

EARLY_STOPPING_PATIENCE = 3

# ==========================
# Mixed Precision
# ==========================

USE_FP16 = True

# ==========================
# Adapter Configuration
# ==========================

ADAPTER_NAME = "business_risk_adapter"

ADAPTER_REDUCTION_FACTOR = 16

NON_LINEARITY = "relu"

# ==========================
# Logging
# ==========================

LOG_LEVEL = "INFO"