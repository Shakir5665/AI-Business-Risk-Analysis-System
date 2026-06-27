import torch

from src.models.business_risk_model import BusinessRiskModel

from configs.model_config import MAX_SEQUENCE_LENGTH

model = BusinessRiskModel(
    num_aspect_classes=3
)

batch_size = 4

input_ids = torch.randint(

    low=0,

    high=250000,

    size=(batch_size, MAX_SEQUENCE_LENGTH)

)

attention_mask = torch.ones(

    batch_size,

    MAX_SEQUENCE_LENGTH,

    dtype=torch.long

)

outputs = model(

    input_ids=input_ids,

    attention_mask=attention_mask

)

print()

print("Sentiment Output")

print(outputs["sentiment_logits"].shape)

print()

print("Aspect Output")

print(outputs["aspect_logits"].shape)