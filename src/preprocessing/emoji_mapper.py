"""
Emoji Mapper - Converts emojis to sentiment tokens
SRS Section: Functional Requirement 18, 19
Decision: 9a (add more emojis), 9b (separate tokens per emoji)
"""

import re
from typing import Dict, List, Tuple

class EmojiMapper:
    """Maps emojis to sentiment tokens as specified in SRS"""
    
    # Comprehensive emoji mapping (positive)
    POSITIVE_EMOJIS = {
        # Smiling faces
        '😊', '😃', '😄', '😆', '😍', '🥰', '😘', '😗', 
        '😙', '😚', '😋', '😛', '😝', '😜',
        '🤪', '😎', '🥳', '🤗', '🤭',
        
        # Hearts and love
        '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💖', '💗', '💓',
        '💞', '💕', '💟', '❣️', '💋', '💥', '🩷', '🩵', '🩶','🎀','💝','🎁',
        
        # Gestures
        '👌', '✌️', '🤞', '🤟', '🤘', '👏', '🙌', '💪', '🎉', '✨', '⭐',
        '🌟', '💫', '⚡', '🔥', '💯', '✅', '✔️', '🎯', '🏆', '👍',
        
        # Thumbs up with skin tones
        '👍🏻', '👍🏼', '👍🏽', '👍🏾', '👍🏿',
    }
    
    # Comprehensive emoji mapping (negative)
    NEGATIVE_EMOJIS = {
        # Angry/frustrated faces
        '😡', '😠', '🤬', '😤', '😈', '👿', '💀', '👹', '👺',
        
        # Sad/crying faces
        '😢', '😭', '😿', '😾', '🙁', '☹️', '😞', '😔', '🥲', '😕', '😣', '😖',
        '😫', '😩', '🥺', '😨', '😰', '😥', '😓', '🤯', '😱', '😪', '🤢', '🤮', '😂','😹',
        '🫡',
        # Negative gestures
        '👎', '💔', '❌', '⭕', '💢', '💨', '🖕', '😒', '🙄', '🤨', '🤔',
        
        # Fear/disgust
        '😨', '😰', '😥', '😓', '😖', '😣', '😫', '😩', '🥺', '🤢', '🤮', '😷',
        '🤒', '🤕', '🦠', '☠️', '⚠️', '🚫', '😏',
        
        # Thumbs down with skin tones
        '👎🏻', '👎🏼', '👎🏽', '👎🏾', '👎🏿',
    }
    
    # Neutral emojis
    NEUTRAL_EMOJIS = {
        '😶', '🤐', '😑', '😬', '🙄', '🤨', '🤔', '🧐', '🤷', '🤷‍♂️', '🤷‍♀️',
        '🤦', '🤦‍♂️', '🤦‍♀️', '🙉', '🙊', '💁', '💁‍♂️', '💁‍♀️', '🙋', '🙋‍♂️', '🙋‍♀️',
        '😐', '😑', '😶', '🤐', '🫤', '🤥',
    }
    
    # Skin tone modifiers (these are not emojis themselves but modify emojis)
    SKIN_TONES = {
        '🏻', '🏼', '🏽', '🏾', '🏿'
    }
    
    def __init__(self):
        """Initialize mapping dictionaries"""
        self.emoji_to_token = {}
        
        # Map positive emojis
        for emoji_char in self.POSITIVE_EMOJIS:
            self.emoji_to_token[emoji_char] = 'positive_emoji'
        
        # Map negative emojis
        for emoji_char in self.NEGATIVE_EMOJIS:
            self.emoji_to_token[emoji_char] = 'negative_emoji'
        
        # Map neutral emojis
        for emoji_char in self.NEUTRAL_EMOJIS:
            self.emoji_to_token[emoji_char] = 'neutral_emoji'
        
        # Handle skin tones
        for tone in self.SKIN_TONES:
            self.emoji_to_token[tone] = 'neutral_emoji'
        
        # FIX: Sort by length descending and use | (OR) pattern
        # This handles multi-character emojis correctly
        all_emojis = sorted(self.emoji_to_token.keys(), key=len, reverse=True)
        pattern = '|'.join(re.escape(e) for e in all_emojis)
        self.emoji_pattern = re.compile(pattern)
        
        print(f"✅ EmojiMapper initialized with {len(self.emoji_to_token)} emojis")
    
    def convert_emojis_to_tokens(self, text: str) -> str:
        """
        Convert each emoji to its sentiment token
        Decision 9b: X - Each emoji becomes separate token
        Example: "good 😊😊" → "good positive_emoji positive_emoji"
        """
        def replace_emoji(match):
            emoji_char = match.group(0)
            token = self.emoji_to_token.get(emoji_char, 'neutral_emoji')
            return f' {token} '  # Add spaces around token
        
        # Replace each emoji individually
        result = self.emoji_pattern.sub(replace_emoji, text)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def get_sentiment_from_emojis(self, text: str) -> str:
        """
        Determine overall sentiment from emojis in text
        Returns: 'positive', 'negative', 'neutral', or None if no emojis
        """
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Find all emojis using the pattern
        for emoji in self.emoji_pattern.findall(text):
            if emoji in self.POSITIVE_EMOJIS:
                positive_count += 1
            elif emoji in self.NEGATIVE_EMOJIS:
                negative_count += 1
            elif emoji in self.NEUTRAL_EMOJIS:
                neutral_count += 1
        
        if positive_count == 0 and negative_count == 0 and neutral_count == 0:
            return None
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_emojis(self, text: str) -> List[Dict]:
        """Extract all emojis with their positions and sentiment"""
        emojis_found = []
        
        # Find all emojis using the pattern
        for match in self.emoji_pattern.finditer(text):
            emoji_char = match.group(0)
            sentiment = 'positive' if emoji_char in self.POSITIVE_EMOJIS else \
                       'negative' if emoji_char in self.NEGATIVE_EMOJIS else 'neutral'
            
            emojis_found.append({
                'emoji': emoji_char,
                'position': match.start(),
                'sentiment': sentiment,
                'token': self.emoji_to_token[emoji_char]
            })
        
        return emojis_found
    
    def get_emoji_stats(self, text: str) -> Dict:
        """Get statistics about emojis in text"""
        emojis = self.extract_emojis(text)
        
        stats = {
            'total_emojis': len(emojis),
            'positive_count': sum(1 for e in emojis if e['sentiment'] == 'positive'),
            'negative_count': sum(1 for e in emojis if e['sentiment'] == 'negative'),
            'neutral_count': sum(1 for e in emojis if e['sentiment'] == 'neutral'),
            'emojis_found': [e['emoji'] for e in emojis],
            'tokens_added': [e['token'] for e in emojis]
        }
        
        return stats


# Test the emoji mapper
if __name__ == "__main__":
    mapper = EmojiMapper()
    
    test_texts = [
        "This product is amazing 😊😊",
        "Very bad quality 😡👎",
        "Mixed feelings 🤔",
        "No emojis here",
        "Multiple emotions 😊😡😐",
        "order kre rgb mouse ekk eth ave normal mouse ekk positive_emoji 🏻 negative_emoji",
        "👍🏻 Thumbs up with skin tone",
        "👎🏽 Thumbs down with dark skin",
        "❤️🩷💕 Love hearts"
    ]
    
    print("=" * 60)
    print("EMOJI MAPPER TEST")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        converted = mapper.convert_emojis_to_tokens(text)
        print(f"Converted: {converted}")
        stats = mapper.get_emoji_stats(text)
        print(f"Emojis found: {stats['emojis_found']}")
        print(f"Tokens added: {stats['tokens_added']}")
        print("-" * 40)