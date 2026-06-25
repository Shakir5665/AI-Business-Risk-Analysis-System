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

        # Project Root
        project_root = Path(__file__).resolve().parents[2]

        dictionary_path = (
            project_root
            / "resources"
            / "slang_dictionary.json"
        )

        if not dictionary_path.exists():
            raise FileNotFoundError(
                f"Slang dictionary not found:\n{dictionary_path}"
            )

        with open(dictionary_path, "r", encoding="utf-8") as f:
            slang_groups = json.load(f)

        self.slang_map = {}

        for standard_word, variations in slang_groups.items():

            # Add the standard word itself
            self.slang_map[standard_word.lower()] = standard_word

            # Add all variations
            for variation in variations:
                self.slang_map[
                    variation.lower()
                ] = standard_word

        logger.info(
            f"Loaded {len(self.slang_map)} slang mappings."
        )

    def normalize(self, text: Optional[str]) -> str:

        if not text:
            return ""

        normalized_words = []

        for word in text.split():

            # Preserve punctuation
            prefix = ""
            suffix = ""

            while len(word) > 0 and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]

            while len(word) > 0 and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]

            normalized = self.slang_map.get(
                word.lower(),
                word
            )

            normalized_words.append(
                prefix + normalized + suffix
            )

        return " ".join(normalized_words)

    def normalize_batch(self, texts):

        return [
            self.normalize(text)
            for text in texts
        ]