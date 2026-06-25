"""
Business Risk Dataset

Custom PyTorch Dataset for multilingual
business review classification.

Responsibilities
----------------
- Read one review
- Tokenize review
- Encode sentiment
- Encode aspects
- Return tensors

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from typing import Dict

import pandas as pd
import torch

from torch.utils.data import Dataset

from src.tokenization.tokenizer import ReviewTokenizer
from src.encoding.sentiment_encoder import SentimentEncoder
from src.encoding.aspect_encoder import AspectEncoder

from src.utils.logger import logger


class BusinessRiskDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: ReviewTokenizer,
        sentiment_encoder: SentimentEncoder,
        aspect_encoder: AspectEncoder
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.sentiment_encoder = sentiment_encoder

        self.aspect_encoder = aspect_encoder

        logger.info(

            f"BusinessRiskDataset initialized with {len(self.dataframe)} samples."

        )

    # --------------------------------------------------

    def __len__(self):

        return len(self.dataframe)

    # --------------------------------------------------

    def __getitem__(
        self,
        index: int
    ) -> Dict:

        row = self.dataframe.iloc[index]

        review_id = row["review_id"]

        review = row["text"]

        sentiment = row["sentiment"]

        aspects = row["aspects"]

        tokens = self.tokenizer.tokenize(

            review

        )

        sentiment = self.sentiment_encoder.encode(

            sentiment

        )

        aspects = self.aspect_encoder.encode(

            aspects

        )

        return {

            "review_id": review_id,

            "text": review,

            "input_ids": tokens["input_ids"],

            "attention_mask": tokens["attention_mask"],

            "sentiment": torch.tensor(

                sentiment,

                dtype=torch.long

            ),

            "aspects": torch.tensor(

                aspects,

                dtype=torch.float

            )

        }