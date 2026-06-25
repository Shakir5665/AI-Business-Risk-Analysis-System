from src.dataset.dataset_loader import DatasetLoader

from src.tokenization.tokenizer import ReviewTokenizer

from src.encoding.sentiment_encoder import SentimentEncoder

from src.encoding.aspect_encoder import AspectEncoder

from src.dataloader.business_dataset import BusinessRiskDataset


loader = DatasetLoader()

train_df = loader.load_train()


dataset = BusinessRiskDataset(

    dataframe=train_df,

    tokenizer=ReviewTokenizer(),

    sentiment_encoder=SentimentEncoder(),

    aspect_encoder=AspectEncoder()

)


print()

print("Dataset Size")

print(len(dataset))

print()

sample = dataset[0]

for key, value in sample.items():

    print(f"{key} :")

    print(value)

    print()