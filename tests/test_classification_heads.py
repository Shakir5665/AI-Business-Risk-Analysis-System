import torch

from src.models.classification_heads import (
    SentimentHead,
    AspectHead
)

# -------------------------------------

dummy_input = torch.randn(
    32,
    768
)

# -------------------------------------

sentiment_head = SentimentHead()

sentiment_output = sentiment_head(
    dummy_input
)

print()

print("Sentiment Output Shape")

print(sentiment_output.shape)

# -------------------------------------

aspect_head = AspectHead(
    num_aspect_classes=3
)

aspect_output = aspect_head(
    dummy_input
)

print()

print("Aspect Output Shape")

print(aspect_output.shape)