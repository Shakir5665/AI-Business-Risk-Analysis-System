import torch

from src.models.backbone import XLMRBackbone

from configs.model_config import (
    MAX_SEQUENCE_LENGTH
)

backbone = XLMRBackbone()

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

cls = backbone(

    input_ids,

    attention_mask

)

print()

print("CLS Shape")

print(cls.shape)

print()

print("Requires Grad")

print(

    any(

        parameter.requires_grad

        for parameter in backbone.backbone.parameters()

    )

)