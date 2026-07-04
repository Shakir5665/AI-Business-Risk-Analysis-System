"""
Model Evaluation Script

Evaluates the trained model on the test dataset.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from src.dataset.dataset_loader import DatasetLoader

from src.tokenization.tokenizer import ReviewTokenizer

from src.encoding.sentiment_encoder import SentimentEncoder

from src.encoding.aspect_encoder import AspectEncoder

from src.dataloader.business_dataset import BusinessRiskDataset

from src.dataloader.data_loader import BusinessDataLoader

from src.models.business_risk_model import BusinessRiskModel

from src.training.checkpoint import CheckpointManager

from src.evaluation.evaluator import Evaluator

from src.utils.device import get_device

# --------------------------------------------------
# Device
# --------------------------------------------------

device = get_device()

# --------------------------------------------------
# Load Test Dataset
# --------------------------------------------------

loader = DatasetLoader()

test_df = loader.load_test()

tokenizer = ReviewTokenizer()

sentiment_encoder = SentimentEncoder()

aspect_encoder = AspectEncoder()

test_dataset = BusinessRiskDataset(

    dataframe=test_df,

    tokenizer=tokenizer,

    sentiment_encoder=sentiment_encoder,

    aspect_encoder=aspect_encoder

)

# --------------------------------------------------
# Test DataLoader
# --------------------------------------------------

data_loader = BusinessDataLoader()

test_loader = data_loader.create_test_loader(

    test_dataset

)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = BusinessRiskModel(

    num_aspect_classes=3

)

model.to(device)

# --------------------------------------------------
# Load Best Model
# --------------------------------------------------

checkpoint_manager = CheckpointManager()

checkpoint = checkpoint_manager.load(

    "/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints/best_model.pt",

    model

)

# --------------------------------------------------
# Evaluator
# --------------------------------------------------

evaluator = Evaluator()

results = evaluator.evaluate(

    model=model,

    dataloader=test_loader,

    device=device

)

# --------------------------------------------------
# Print Results
# --------------------------------------------------

print()

print("=" * 60)

print("FINAL TEST RESULTS")

print("=" * 60)

print()

print("Loss")

print(f"  Sentiment : {results['loss']['sentiment']:.4f}")

print(f"  Aspect    : {results['loss']['aspect']:.4f}")

print(f"  Total     : {results['loss']['total']:.4f}")

print()

print("Sentiment")

print(f"  Accuracy  : {results['sentiment']['accuracy']:.4f}")

print(f"  Precision : {results['sentiment']['precision']:.4f}")

print(f"  Recall    : {results['sentiment']['recall']:.4f}")

print(f"  F1 Score  : {results['sentiment']['f1']:.4f}")

print()

print("Aspect")

print(f"  Micro F1  : {results['aspect']['micro_f1']:.4f}")

print(f"  Macro F1  : {results['aspect']['macro_f1']:.4f}")

print()

print("=" * 60)

from src.evaluation.confusion_matrix import (
    ConfusionMatrixGenerator
)

cm = ConfusionMatrixGenerator()

cm.generate(

    true_labels=results["targets"]["sentiment"].numpy(),

    predicted_labels=results["predictions"]["sentiment"].numpy(),

    class_names=[

        "Negative",

        "Neutral",

        "Positive"

    ],

    save_path="outputs/figures/confusion_matrix.png"

)

print()

print("Confusion matrix saved.")