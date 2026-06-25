"""
Slang Normalizer

Normalizes common internet abbreviations and chat slang.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import re
from typing import Optional

from src.utils.logger import logger


class SlangNormalizer:

    def __init__(self):

        self.slang_map = {

            # General
            "plz": "please",
            "pls": "please",
            "plss": "please",

            "thx": "thanks",
            "tnx": "thanks",
            "thanx": "thanks",

            "u": "you",
            "ur": "your",
            "urs": "yours",

            "bcz": "because",
            "bcz": "because",
            "coz": "because",
            "cozz": "because",

            "msg": "message",

            "omg": "oh my god",

            "idk": "i do not know",

            "imo": "in my opinion",

            "btw": "by the way",

            "asap": "as soon as possible",

            "faq": "frequently asked questions",

            "approx": "approximately",

            "abt": "about",

            "tmrw": "tomorrow",

            "tomo": "tomorrow",

            "luv": "love",

            "gr8": "great",

            "g8": "great",

            "b4": "before",

            "w8": "wait",

            "gud": "good",

            "okies": "okay",
            "oky": "okay",
            "okkk": "okay",

            "sry": "sorry",
            "soz": "sorry",

            "np": "no problem",

            "ty": "thank you",

            "tysm": "thank you so much",

            "wtf": "what the",
        }

        logger.info(
            f"Loaded {len(self.slang_map)} slang mappings."
        )

    def normalize(self, text: Optional[str]) -> str:

        if not text:
            return ""

        words = text.split()

        normalized = []

        for word in words:

            key = word.lower()

            normalized.append(

                self.slang_map.get(
                    key,
                    word
                )

            )

        return " ".join(normalized)

    def normalize_batch(self, texts):

        return [

            self.normalize(text)

            for text in texts

        ]