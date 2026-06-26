"""
Model Configuration

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

# --------------------------------------------------
# Base Model
# --------------------------------------------------

MODEL_NAME = "xlm-roberta-base"

# --------------------------------------------------
# Tokenization
# --------------------------------------------------

MAX_SEQUENCE_LENGTH = 128

# --------------------------------------------------
# XLM-R Architecture
# --------------------------------------------------

HIDDEN_SIZE = 768

# --------------------------------------------------
# Adapter Configuration
# --------------------------------------------------

ADAPTER_DIM = 64

ADAPTER_DROPOUT = 0.1

# --------------------------------------------------
# Classification Heads
# --------------------------------------------------

CLASSIFIER_DROPOUT = 0.3

NUM_SENTIMENT_CLASSES = 3