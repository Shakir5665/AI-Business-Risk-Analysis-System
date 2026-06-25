"""
Pipeline Components

Provides all reusable machine learning
pipeline components.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from src.encoding.sentiment_encoder import SentimentEncoder
from src.encoding.aspect_encoder import AspectEncoder
from src.tokenization.tokenizer import ReviewTokenizer
from src.utils.logger import logger


class PipelineComponents:

    """
    Centralized access to all reusable
    ML pipeline components.
    """

    def __init__(self):

        logger.info(
            "Initializing Pipeline Components..."
        )

        self.sentiment_encoder = SentimentEncoder()

        self.aspect_encoder = AspectEncoder()

        self.tokenizer = ReviewTokenizer()

        logger.info(
            "Pipeline Components Ready."
        )