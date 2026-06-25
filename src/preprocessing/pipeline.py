"""
Preprocessing Pipeline

Executes the complete preprocessing pipeline.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

from typing import List

from src.utils.logger import logger

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.emoji_mapper import EmojiMapper
from src.preprocessing.sinhala_converter import SinhalaConverter
from src.preprocessing.phonetic_normalizer import PhoneticNormalizer
from src.preprocessing.slang_normalizer import SlangNormalizer
from src.preprocessing.repeat_normalizer import RepeatNormalizer


class PreprocessingPipeline:

    """
    Complete preprocessing pipeline.

    Order:

    1. Basic Cleaning
    2. Emoji Mapping
    3. Sinhala → Singlish
    4. Phonetic Normalization
    5. Slang Normalization
    6. Repeated Character Normalization
    """

    def __init__(self):

        logger.info("Initializing preprocessing pipeline...")

        self.cleaner = TextCleaner()

        self.emoji_mapper = EmojiMapper()

        self.sinhala_converter = SinhalaConverter()

        self.phonetic_normalizer = PhoneticNormalizer()

        self.slang_normalizer = SlangNormalizer()

        self.repeat_normalizer = RepeatNormalizer()

        logger.info("Preprocessing pipeline initialized.")

    def process(self, text: str) -> str:

        if not text:
            return ""

        # --------------------------------------------------

        text = self.cleaner.clean(text)

        # --------------------------------------------------

        text = self.emoji_mapper.convert_emojis_to_tokens(text)

        # --------------------------------------------------

        if self.sinhala_converter.detect_sinhala(text):

            text = self.sinhala_converter.convert_to_singlish(text)

        # --------------------------------------------------

        text = self.phonetic_normalizer.normalize(text)

        # --------------------------------------------------

        text = self.slang_normalizer.normalize(text)

        # --------------------------------------------------

        text = self.repeat_normalizer.normalize(text)

        # --------------------------------------------------

        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:

        return [

            self.process(text)

            for text in texts

        ]