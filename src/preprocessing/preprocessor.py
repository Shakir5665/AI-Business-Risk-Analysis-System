"""
Review Preprocessor

Single entry point for preprocessing.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

from typing import List

from src.utils.logger import logger

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.emoji_mapper import EmojiMapper
from src.preprocessing.repeat_normalizer import RepeatNormalizer
from src.preprocessing.srilankan_normalizer import SriLankanNormalizer


class ReviewPreprocessor:

    def __init__(self):

        logger.info("Initializing ReviewPreprocessor...")

        self.cleaner = TextCleaner()

        self.emoji_mapper = EmojiMapper()

        self.repeat = RepeatNormalizer()

        self.srilankan = SriLankanNormalizer()

        logger.info("ReviewPreprocessor initialized successfully.")

    # ---------------------------------------------------------

    def preprocess(self, text: str) -> str:

        if text is None:
            return ""

        text = self.cleaner.clean(text)

        text = self.emoji_mapper.convert_emojis_to_tokens(text)

        text = self.repeat.normalize(text)

        text = self.srilankan.normalize(text)

        return text.strip()

    # ---------------------------------------------------------

    def preprocess_batch(
        self,
        texts: List[str]
    ) -> List[str]:

        return [

            self.preprocess(text)

            for text in texts

        ]