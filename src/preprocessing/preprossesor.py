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
from src.preprocessing.dictionary_normalizer import DictionaryNormalizer
from src.preprocessing.repeat_normalizer import RepeatNormalizer
from src.preprocessing.phonetic_normalizer import PhoneticNormalizer
from src.preprocessing.slang_normalizer import SlangNormalizer


class ReviewPreprocessor:

    def __init__(self):

        logger.info("Initializing ReviewPreprocessor...")

        self.cleaner = TextCleaner()

        self.emoji_mapper = EmojiMapper()

        self.dictionary = DictionaryNormalizer()

        self.repeat = RepeatNormalizer()

        self.phonetic = PhoneticNormalizer()

        self.slang = SlangNormalizer()

        logger.info("ReviewPreprocessor initialized successfully.")

    # ---------------------------------------------------------

    def preprocess(self, text: str) -> str:

        if text is None:
            return ""

        text = self.cleaner.clean(text)

        text = self.emoji_mapper.convert_emojis_to_tokens(text)

        text = self.dictionary.normalize(text)

        text = self.repeat.normalize(text)

        text = self.phonetic.normalize(text)

        text = self.slang.normalize(text)

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