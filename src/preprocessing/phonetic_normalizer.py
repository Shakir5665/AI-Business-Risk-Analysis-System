"""
Phonetic Normalizer - Normalizes spelling variations
SRS Section: Objective 2 - Phonetic mapping
Decision 11: A - Dictionary-based
"""

import re
from typing import Dict, List, Optional

class PhoneticNormalizer:
    """
    Normalizes phonetic variations in Singlish text
    Example: "hodaai" → "hodai", "delivari" → "delivery"
    """
    
    def __init__(self):
        """Initialize the normalization dictionary"""
        self.normalization_map = self._build_normalization_map()
        self.reverse_map = self._build_reverse_map()
    
    def _build_normalization_map(self) -> Dict[str, str]:
        """Build dictionary of variation → standard form"""
        
        return {
    # Delivery variations
    'delivery': 'delivery',
    'delivary': 'delivery',
    'delivari': 'delivery',
    'delivry': 'delivery',
    'dilivery': 'delivery',
    'dilivary': 'delivery',
    'dilivari': 'delivery',
    'dilivry': 'delivery',
    'deliwary': 'delivery',
    
    # Quality variations
    'quality': 'quality',
    'qualiti': 'quality',
    'quallity': 'quality',
    'kwality': 'quality',
    'kwaliti': 'quality',
    'qulity': 'quality',
    'qualty': 'quality',
    
    # Product variations
    'product': 'product',
    'produkt': 'product',
    'prodact': 'product',
    'produt': 'product',
    'product ek': 'product',
    'produk': 'product',
    
    # Good variations
    'hodai': 'හොඳයි',
    'hoda': 'හොඳයි',
    'hodaai': 'හොඳයි',
    'hodayi': 'හොඳයි',
    'hodaayi': 'හොඳයි',
    'hodaii': 'හොඳයි',
    'hodath': 'හොඳයි',
    'hody' : 'හොඳයි',
    "eka": "එක",
    'ek':'එක',
    
    # Bad / Not good variations
    "narakai": "නරකයි",
    "naraka": "නරකයි",
    "narakay": "නරකයි",
    "narakaii": "නරකයි",
    "nadda": "නැද්ද",
    "na": "නැ",
    "neh": "නැ",
    "ne":"නැ",
    "naha": "නැහැ",
    "netha": "නැත",
    "nae": "නෑ",
    "nemei": "නෙමෙයි",
    
    # Late variations
    "pramada" : "ප්‍රමාද",
    "pramadai": "ප්‍රමාද",
    "pramadae": "ප්‍රමාද",
    "pramaday": "ප්‍රමාද",
    "pramada": "ප්‍රමාද",
    "prkku": "ප්‍රමාද",
    "unaa" : "වුණා",
    "una"  : "වුණා",
    "un" : "වුණා",

    
    # Trust variations
   "viswasaya": "විශ්වාසය",
    "visvasa": "විශ්වාසය",
    "viswasay": "විශ්වාසය",
    "viswasa": "විශ්වාසය",
    "withwasaya": "විශ්වාසය",
    "wishwasa": "විශ්වාසය",
    
    # Price variations
    'price': 'price',
    'pric': 'price',
    'praice': 'price',
    'pris': 'price',
    
    # Service variations
    'service': 'service',
    'servis': 'service',
    'sarvis': 'service',
    'sarvice': 'service',
    'sarvice': 'service',
    'srvc' : 'service',

    
    # Time variations
    'time': 'time',
    'taim': 'time',
    'tyme': 'time',
    'taim ': 'time',
    
    # Money variations
    'money': 'money',
    'mone': 'money',
    'muni': 'money',
    'maney': 'money',
    'salli': 'money',
    'gaan': 'money',
    
    # Return variations
    'return': 'return',
    'retun': 'return',
    'retarn': 'return',
    'riturn': 'return',
    'rifnd': 'return',
    'rfnd':'refund',

    
    # Order variations
    'order': 'order',
    'orda': 'order',
    'ordar': 'order',
    'oder': 'order',
    'odr': 'order',
    
    # Payment variations
    'payment': 'payment',
    'paymnt': 'payment',
    'paymen': 'payment',
    'peyment': 'payment',
    "pmnt": "payment",

    #Please variations
    'pls' : 'please',
    'plz': 'please',
    'pleeaassee': 'please',
    'pleeees': 'please',
    'plees': 'please',
    'pleez': 'please',
    'pleaze': 'please',
    'bleez' : 'please',
    
    # Working / Functioning
    "wada": "වැඩ",
    "vaed": "වැඩ",
    "veda": "වැඩ",
    "wedak": "වැඩක්",
    "weda": "වැඩ",
    "krnne": "කරන්නේ",
    "karana" : "කරන්නේ",
    'karanna' : "කරන්නේ",
    'karan' : "කරන්නේ",
    'krn' : "කරන්නේ",
    'karen' : "කරන්නේ",
    
    
    # Fake / Scam
    'fake': 'fake',
    'feak': 'fake',
    'facke': 'fake',
    'boru': 'fake',
    'boruwak': 'fake',
    'original nemeyi': 'fake',
    'copy': 'fake',
    'duplicate': 'fake',
    
    # Delivery issues
    'damage': 'damage',
    'damage wela': 'damage',
    'hirila': 'damage',
    'kaduna': 'damage',
    'bindila': 'damage',
    'kaedunaa': 'damage',
    

    
    # Response issue
    'reply na': 'reply',
    'riplyi ': 'reply',
    'rply': 'reply',
    'rspnse': 'response',
    'respnse': 'response',
    
    
    # Common Sinhala-English mixed words
    "eka": "එක",
    "ekak": "එකක්",
    "thiyenawa": "තියෙනවා",
    "nadda": "නැද්ද",
    "match": "ගැළපෙනවා",
    "mach": "ගැළපෙනවා",
    "hodata": "හොඳට",
    "harida": "හරිද",
    "hari": "හරි",
    "hri": "හරි",
    "hariyata": "හරි",
    "harima": "හරිම",
    "gdak": "ගොඩක්",
    "aththtm": "ඇත්තටම",
    "sirawat": "ඇත්තටම",
    "supri": "සුපිරි",
    "ganna": "ගන්න",
    "gnn": "ගන්න",
    "denn": "දෙන්න",
    "yanna": "යන්න",
    "enna": "එන්න",
    "krnn": "කරන්න",
    "krn": "කරන්න",
    "hambuna": "හම්බුනා",
    "lebuna": "හම්බුනා",
    "labuna": "හම්බුනා",
    "awe": "ආවේ",
    "awa": "ආවා",
    "evva": "ආවේ",
    "ewala": "ආවේ",
    "thx": "ස්තුතියි",
    "niymyi": "නියමයි",
    "laabyi": "ලාභයි",
    "oodr": "ඕඩර්",
    
    # Warranty
    'warranty': 'warranty',
    'woranti': 'warranty',
    'wrnti': 'warranty'
    
}
    
    def _build_reverse_map(self) -> Dict[str, List[str]]:
        """Build reverse mapping for debugging"""
        reverse = {}
        for variation, standard in self.normalization_map.items():
            if standard not in reverse:
                reverse[standard] = []
            reverse[standard].append(variation)
        return reverse
    
    def normalize(self, text: str) -> str:
        """
        Normalize phonetic variations in text
        
        Args:
            text: Input text with possible variations
            
        Returns:
            Text with normalized spellings
        """
        if not text:
            return text
        
        result = text.lower()
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_variations = sorted(
            self.normalization_map.keys(),
            key=len,
            reverse=True
        )
        
        # Apply replacements
        for variation in sorted_variations:
            standard = self.normalization_map[variation]
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(variation) + r'\b'
            result = re.sub(pattern, standard, result)
        
        return result
    
    def add_custom_mapping(self, variation: str, standard: str):
        """Add custom mapping for domain-specific terms"""
        self.normalization_map[variation] = standard
        if standard not in self.reverse_map:
            self.reverse_map[standard] = []
        self.reverse_map[standard].append(variation)
    
    def get_variations(self, standard_word: str) -> List[str]:
        """Get all variations for a standard word"""
        return self.reverse_map.get(standard_word, [])
    
    def explain_normalization(self, text: str) -> Dict:
        """
        Explain what changes were made during normalization
        Returns dict with original, normalized, and changes made
        """
        original = text
        normalized = self.normalize(text)
        
        changes = []
        if original != normalized:
            # Find what changed
            original_words = original.lower().split()
            normalized_words = normalized.split()
            
            for i, (orig, norm) in enumerate(zip(original_words, normalized_words)):
                if orig != norm:
                    changes.append({
                        'position': i,
                        'original': orig,
                        'normalized': norm,
                        'type': 'phonetic'
                    })
        
        return {
            'original': original,
            'normalized': normalized,
            'changes_made': len(changes) > 0,
            'changes': changes
        }


# Test the normalizer
if __name__ == "__main__":
    normalizer = PhoneticNormalizer()
    
    test_phrases = [
        "delivari is very late",
        "hodaai product quality super",
        "viswasa issue ekak thiyenawa",
        "delivary eka pramadai",
        "this is a produkt with good kwaliti",
    ]
    
    print("=" * 60)
    print("PHONETIC NORMALIZER TEST")
    print("=" * 60)
    
    for phrase in test_phrases:
        result = normalizer.normalize(phrase)
        print(f"\nOriginal: {phrase}")
        print(f"Normalized: {result}")
        
        # Show explanation
        explanation = normalizer.explain_normalization(phrase)
        if explanation['changes_made']:
            print(f"Changes: {explanation['changes']}")
        print("-" * 40)