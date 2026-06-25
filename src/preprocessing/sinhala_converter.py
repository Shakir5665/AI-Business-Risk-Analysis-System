"""
Sinhala Converter

Converts Sinhala reviews into normalized Singlish/English.

Pipeline

Sentence
    │
    ▼
Word Tokenizer
    │
    ▼
Dictionary Lookup
    │
    ▼
Unknown Words
    │
    ▼
Fallback Transliteration
    │
    ▼
Normalized Sentence

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import json
import re
from pathlib import Path
from typing import List

from src.utils.logger import logger


class SinhalaConverter:

    def __init__(self):

        self.dictionary = self._load_dictionary()

        self.sinhala_pattern = re.compile(r'[\u0D80-\u0DFF]')

        logger.info(
            f"SinhalaConverter loaded {len(self.dictionary)} dictionary words."
        )

    # -----------------------------------------------------
    # Load Dictionary
    # -----------------------------------------------------

    def _load_dictionary(self):

        dictionary_path = Path(
            "resources/sinhala_dictionary.json"
        )

        if not dictionary_path.exists():

            logger.warning(
                "Sinhala dictionary not found."
            )

            return {}

        with open(
            dictionary_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # -----------------------------------------------------
    # Detect Sinhala
    # -----------------------------------------------------

    def detect_sinhala(self, text: str) -> bool:

        if not text:

            return False

        return bool(
            self.sinhala_pattern.search(text)
        )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    def tokenize(self, text: str) -> List[str]:

        return text.split()

    # -----------------------------------------------------
    # Dictionary Lookup
    # -----------------------------------------------------

    def lookup(self, word: str):

        return self.dictionary.get(word)

    # -----------------------------------------------------
    # Convert Sentence
    # -----------------------------------------------------

    def convert_to_singlish(self, text: str):

        if not text:

            return ""

        words = self.tokenize(text)

        converted = []

        for word in words:

            mapped = self.lookup(word)

            if mapped:

                converted.append(mapped)

            else:

                converted.append(
                    self.fallback_convert(word)
                )

        return " ".join(converted)

    # -----------------------------------------------------
    # Fallback
    # (Implemented in Part 2)
    # -----------------------------------------------------

    def fallback_convert(self, word: str):

        return word

    # -----------------------------------------------------

    def convert_batch(self, texts):

        return [

            self.convert_to_singlish(text)

            for text in texts

        ]