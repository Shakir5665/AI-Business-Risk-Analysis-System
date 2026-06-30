"""
Training Script

Runs complete model training.

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

from src.training.optimizer import OptimizerFactory

from src.training.scheduler import SchedulerFactory

from src.training.trainer import Trainer

from src.utils.device import get_device

from configs.training_config import NUM_EPOCHS


# --------------------------------------------------
# Device
# --------------------------------------------------

device = get_device()

# --------------------------------------------------
# Dataset
# --------------------------------------------------

loader = DatasetLoader()

train_df = loader.load_train()

validation_df = loader.load_validation()

tokenizer = ReviewTokenizer()

sentiment_encoder = SentimentEncoder()

aspect_encoder = AspectEncoder()

train_dataset = BusinessRiskDataset(

    dataframe=train_df,

    tokenizer=tokenizer,

    sentiment_encoder=sentiment_encoder,

    aspect_encoder=aspect_encoder

)

validation_dataset = BusinessRiskDataset(

    dataframe=validation_df,

    tokenizer=tokenizer,

    sentiment_encoder=sentiment_encoder,

    aspect_encoder=aspect_encoder

)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

data_loader = BusinessDataLoader()

train_loader = data_loader.create_train_loader(

    train_dataset

)

validation_loader = data_loader.create_validation_loader(

    validation_dataset

)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = BusinessRiskModel(

    num_aspect_classes=3

)

model.to(device)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = OptimizerFactory.create(

    model

)

# --------------------------------------------------
# Scheduler
# --------------------------------------------------

total_training_steps = (

    len(train_loader)

    * NUM_EPOCHS

)

scheduler = SchedulerFactory.create(

    optimizer,

    total_training_steps

)

# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(

    model=model,

    train_loader=train_loader,

    validation_loader=validation_loader,

    optimizer=optimizer,

    scheduler=scheduler,

    device=device

)

# --------------------------------------------------
# Start Training
# --------------------------------------------------

trainer.train()