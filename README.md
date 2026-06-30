$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_preprocessor.py

# AI-Business-Risk-Analysis-System

## Folder Structure

```text
AI-Business-Risk-Analysis-System/
├── checkpoints/                          # Saved training checkpoints
├── configs/                              # Configuration modules
│   ├── config.py                         # Main config loader
│   ├── model_config.py                  # Model architecture settings
│   ├── model_labels.py                  # Label definitions
│   ├── paths.py                         # File and directory paths
│   ├── preprocessing_config.py         # Preprocessing settings
│   ├── storage.py                       # Storage configuration
│   └── training_config.py               # Training parameters
├── data/                                 # Dataset files
│   ├── final/                            # Final prepared dataset
│   │   └── dataset.jsonl                 # Final JSONL dataset
│   ├── processed/                        # Processed splits
│   │   ├── test.jsonl                    # Test split
│   │   ├── train.jsonl                   # Training split
│   │   └── validation.jsonl              # Validation split
│   └── raw/                              # Raw source data
├── models/                               # Model assets and tokenizer files
│   ├── encoders/                         # Encoder checkpoints
│   └── tokenizer/                        # Tokenizer resources
│       ├── sentencepiece.bpe.model
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── notebooks/                            # Jupyter notebooks
│   └── train.ipynb                       # Training notebook
├── outputs/                              # Generated outputs
│   ├── figures/                          # Plots and charts
│   ├── logs/                             # Training/inference logs
│   └── reports/                          # Evaluation reports
├── resources/                            # Supporting assets
│   └── slang_dictionary.json             # Slang mapping dictionary
├── src/                                  # Main implementation package
│   ├── __init__.py
│   ├── analysis/                         # Statistical and visualization helpers
│   │   ├── statistics.py
│   │   └── visualization.py
│   ├── dataloader/                       # Data-loading pipeline
│   │   ├── __init__.py
│   │   ├── business_dataset.py
│   │   └── data_loader.py
│   ├── dataset/                          # Dataset utilities and loaders
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── dataset_loader.py
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   └── validator.py
│   ├── encoding/                         # Encoder modules
│   │   ├── __init__.py
│   │   ├── aspect_encoder.py
│   │   └── sentiment_encoder.py
│   ├── evaluation/                       # Evaluation code
│   │   └── __init__.py
│   ├── inference/                        # Prediction/inference code
│   │   └── __init__.py
│   ├── models/                           # Model definitions
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── business_risk_model.py
│   │   └── classification_heads.py
│   ├── preprocessing/                    # Text preprocessing modules
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── emoji_mapper.py
│   │   ├── preprocessor.py
│   │   ├── repeat_normalizer.py
│   │   └── srilankan_normalizer.py
│   ├── tokenization/                     # Tokenizer implementation
│   │   ├── __init__.py
│   │   └── tokenizer.py
│   ├── training/                         # Training pipeline code
│   │   └── __init__.py
│   ├── utils/                            # Shared helpers
│   │   ├── __init__.py
│   │   ├── device.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   └── __init__.py
│   └── visualization/                    # Visualization helpers
│       └── __init__.py
├── tests/                                # Test suite
│   ├── __init__.py
│   ├── test_adapter.py
│   ├── test_aspect_encoder.py
│   ├── test_aspect_load.py
│   ├── test_business_dataset.py
│   ├── test_classification_heads.py
│   ├── test_dataloader.py
│   ├── test_dataset_loader.py
│   ├── test_dataset_splitter.py
│   ├── test_preprocessor.py
│   ├── test_sentiment_encoder.py
│   ├── test_sentiment_load.py
│   └── test_tokenizer.py
├── experiments.py                        # Experiment workflow script
├── predict.py                            # Prediction script
├── requirements.txt                      # Python dependencies
├── test_setup.py                         # Environment and setup validation script
├── train.py                              # Training script
├── walkthrough.md                        # Detailed walkthrough guide
└── README.md                             # Project overview and documentation
```

$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_preprocessor.py