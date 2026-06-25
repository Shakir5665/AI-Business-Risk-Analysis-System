"""
Training Configuration

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

# --------------------------------------------------
# Dataset Split
# --------------------------------------------------

TRAIN_SIZE = 0.80

VALIDATION_SIZE = 0.10

TEST_SIZE = 0.10

# --------------------------------------------------
# Random Seed
# --------------------------------------------------

RANDOM_STATE = 42

# --------------------------------------------------
# DataLoader
# --------------------------------------------------

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = False

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False