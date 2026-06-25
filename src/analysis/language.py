"""
Language Analysis Engine

Estimates language distribution in the dataset.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import re
from collections import Counter


class LanguageAnalyzer:

    SINHALA_PATTERN = re.compile(r'[\u0D80-\u0DFF]')

    COMMON_SINGLISH = {
        "eka",
        "hari",
        "hodai",
        "naha",
        "mage",
        "oya",
        "mata",
        "mama",
        "ganna",
        "karanna",
        "thiyenawa",
        "thibba",
        "mokak",
        "nathi",
        "kohomada",
        "godak",
        "lassanai",
        "wada",
        "wenawa",
        "delivery",
        "daraz"
    }

    def analyze(self, dataset):

        distribution = Counter()

        for sample in dataset:

            text = sample["text"].lower()

            # Sinhala Unicode
            has_sinhala = bool(
                self.SINHALA_PATTERN.search(text)
            )

            words = text.split()

            english = 0
            singlish = 0

            for word in words:

                if word in self.COMMON_SINGLISH:
                    singlish += 1

                elif word.isascii():
                    english += 1

            # Decision

            if has_sinhala and english > 0:
                distribution["Mixed"] += 1

            elif has_sinhala:
                distribution["Sinhala"] += 1

            elif singlish > english:
                distribution["Singlish"] += 1

            else:
                distribution["English"] += 1

        return dict(distribution)