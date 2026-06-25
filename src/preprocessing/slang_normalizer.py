"""
Slang Normalizer

Loads slang mappings from JSON.

Project:
AI-Powered Business Risk Analysis and Recommendation System 
"""

import json
from pathlib import Path
from typing import Optional

from src.utils.logger import logger


class SlangNormalizer:

    def __init__(self):

        dictionary_path = Path("resources/slang_dictionary.json")

        with open(dictionary_path, "r", encoding="utf-8") as f:

            slang_groups = json.load(f)

        self.slang_map = {}

        for standard_word, variations in slang_groups.items():

            self.slang_map[standard_word] = standard_word

            for variation in variations:

                self.slang_map[variation.lower()] = standard_word

        logger.info(
            f"Loaded {len(self.slang_map)} slang mappings."
        )

    def normalize(self, text: Optional[str]) -> str:

        if not text:
            return ""

        words = text.split()

        normalized_words = []

        for word in words:

            key = word.lower()

            normalized_words.append(

                self.slang_map.get(
                    key,
                    word
                )

            )

        return " ".join(normalized_words)

    def normalize_batch(self, texts):

        return [

            self.normalize(text)

            for text in texts

        ]