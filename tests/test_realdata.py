"""
Interactive Testing Script with Real Data

Allows manual input of review comments to predict sentiment and business aspects.
Designed to run either in Google Colab or locally.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tokenization.tokenizer import ReviewTokenizer
from src.encoding.sentiment_encoder import SentimentEncoder
from src.encoding.aspect_encoder import AspectEncoder
from src.models.business_risk_model import BusinessRiskModel
from src.training.checkpoint import CheckpointManager
from src.utils.device import get_device

def main():
    print("=" * 60)
    print("  AI-Powered Business Risk Analysis & Recommendation System")
    print("             Interactive Prediction CLI Tool")
    print("=" * 60)

    # 1. Device Setup
    device = get_device()

    # 2. Tokenizers and Encoders
    print("\n[1/3] Initializing Tokenizer and Encoders...")
    try:
        tokenizer = ReviewTokenizer()
        sentiment_encoder = SentimentEncoder()
        aspect_encoder = AspectEncoder()
        print("Initialization successful.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # 3. Model Setup
    print("\n[2/3] Building Business Risk Model...")
    try:
        model = BusinessRiskModel(
            num_aspect_classes=len(aspect_encoder.classes)
        )
        model.to(device)
        print("Model architecture built successfully.")
    except Exception as e:
        print(f"Error building model architecture: {e}")
        return

    # 4. Checkpoint Loading
    print("\n[3/3] Loading Model Checkpoint...")
    
    colab_path = "/content/AI-Business-Risk-Analysis-System/checkpoints/best_model.pt"
    local_path = project_root / "checkpoints" / "best_model.pt"
    
    selected_path = colab_path
    
    # Check if files exist at defaults
    if Path(colab_path).exists():
        selected_path = colab_path
        print(f"Found checkpoint at Colab path: {colab_path}")
    elif Path(local_path).exists():
        selected_path = str(local_path)
        print(f"Found checkpoint at local path: {selected_path}")
    else:
        print(f"Checkpoint not found at default locations:")
        print(f"  - Colab Path: {colab_path}")
        print(f"  - Local Path: {local_path}")
        print("Please enter a custom path, or press Enter to default to Colab path.")
        user_input_path = input("Checkpoint path: ").strip()
        if user_input_path:
            selected_path = user_input_path
        else:
            selected_path = colab_path

    checkpoint_manager = CheckpointManager()
    
    try:
        print(f"Attempting to load weights from: {selected_path} ...")
        checkpoint_manager.load(selected_path, model)
        model.to(device)
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"\nWarning: Could not load checkpoint weights. Reason: {e}")
        print("If you are preparing this for Google Colab, you can ignore this warning.")
        print("The script will now proceed, but predictions may be random/untrained.")
        print("Press Enter to continue, or Ctrl+C to abort.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\nAborting...")
            return

    model.eval()

    # 5. Interactive loop
    print("\n" + "=" * 60)
    print("                      READY FOR PREDICTIONS")
    print("  Type your review comment and press Enter to predict.")
    print("  Type 'exit' or 'quit' to close the program.")
    print("=" * 60 + "\n")

    while True:
        try:
            comment = input("Review Comment: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not comment:
            continue

        if comment.lower() in ["exit", "quit"]:
            print("Exiting interactive testing. Goodbye!")
            break

        # Run Prediction
        try:
            # Tokenize comment
            tokens = tokenizer.tokenize(comment)
            
            # Prepare batch of size 1 and move to device
            input_ids = tokens["input_ids"].unsqueeze(0).to(device)
            attention_mask = tokens["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Sentiment processing
                sentiment_logits = outputs["sentiment_logits"]
                sentiment_probs = F.softmax(sentiment_logits, dim=1)[0]
                sentiment_pred_idx = torch.argmax(sentiment_probs).item()
                sentiment_label = sentiment_encoder.decode(sentiment_pred_idx)
                sentiment_confidence = sentiment_probs[sentiment_pred_idx].item() * 100

                # Aspect processing
                aspect_logits = outputs["aspect_logits"]
                aspect_probs = torch.sigmoid(aspect_logits)[0]
                
                # Active aspects (threshold = 0.5)
                active_aspects = []
                aspect_details = []
                for idx, aspect_name in enumerate(aspect_encoder.classes):
                    prob = aspect_probs[idx].item()
                    is_active = prob >= 0.5
                    aspect_details.append((aspect_name, prob * 100, is_active))
                    if is_active:
                        active_aspects.append(aspect_name)

            # Print Output
            print("\n" + "-" * 50)
            print("                     PREDICTION")
            print("-" * 50)
            print(f"Sentiment: {sentiment_label.capitalize()} ({sentiment_confidence:.2f}% confidence)")
            print("\nAspects Detected:")
            if active_aspects:
                for aspect in active_aspects:
                    print(f"  * {aspect.capitalize()}")
            else:
                print("  (No aspects detected with probability >= 50%)")
            
            print("\nDetailed Probabilities:")
            print("  Sentiment:")
            for idx, label in enumerate(sentiment_encoder.classes):
                print(f"    - {label.capitalize():<10}: {sentiment_probs[idx].item() * 100:.2f}%")
            
            print("  Aspects:")
            for aspect_name, prob, is_active in aspect_details:
                status = "Active" if is_active else "Inactive"
                print(f"    - {aspect_name.capitalize():<10}: {prob:.2f}% [{status}]")
            
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"\nAn error occurred during prediction: {e}\n")

if __name__ == "__main__":
    main()
