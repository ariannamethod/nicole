"""
Nicole Bootstrap — Phrase Shapes
Utilities for matching/selecting phrase structures

STUB: Will guide sentence structure selection
"""

def match_shape(sentence, shape_pattern):
    """
    Check if sentence matches phrase shape

    Args:
        sentence: str
        shape_pattern: dict with "pattern" and "length"

    Returns:
        bool: True if matches
    """
    words = sentence.split()
    expected_length = shape_pattern.get("length", 10)

    # Allow ±20% length variance
    if not (0.8 * expected_length <= len(words) <= 1.2 * expected_length):
        return False

    # Check first/last word match (simplified)
    pattern = shape_pattern.get("pattern", "")
    if "..." in pattern:
        parts = pattern.split(" ... ")
        if len(parts) == 2:
            first_word, last_word = parts
            if len(words) >= 2:
                if words[0].lower() != first_word or words[-1].lower() != last_word:
                    return False

    return True
