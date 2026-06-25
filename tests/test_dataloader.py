from src.dataset.dataset_loader import DatasetLoader

from src.tokenization.tokenizer import ReviewTokenizer

from src.encoding.sentiment_encoder import SentimentEncoder

from src.encoding.aspect_encoder import AspectEncoder

from src.dataloader.business_dataset import BusinessRiskDataset

from src.dataloader.data_loader import BusinessDataLoader


loader = DatasetLoader()

train_df = loader.load_train()


dataset = BusinessRiskDataset(

    dataframe=train_df,

    tokenizer=ReviewTokenizer(),

    sentiment_encoder=SentimentEncoder(),

    aspect_encoder=AspectEncoder()

)

train_loader = BusinessDataLoader().create_train_loader(

    dataset

)

batch = next(

    iter(train_loader)

)

print()

print("Batch Keys")

print(batch.keys())

print()

print("Input IDs Shape")

print(batch["input_ids"].shape)

print()

print("Attention Mask Shape")

print(batch["attention_mask"].shape)

print()

print("Sentiment Shape")

print(batch["sentiment"].shape)

print()

print("Aspect Shape")

print(batch["aspects"].shape)