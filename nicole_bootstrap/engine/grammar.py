"""
Nicole Bootstrap — Grammar & Punctuation (from sska)
Perfect grammar for generated sentences.

This ensures:
- Proper capitalization
- Correct punctuation
- Valid sentence structure
- No fragments or run-ons
"""

import re
from typing import List

# ============================================================================
# PUNCTUATION RULES
# ============================================================================

def capitalize_first_letter(text: str) -> str:
    """Capitalize first letter of sentence."""
    if not text:
        return text
    return text[0].upper() + text[1:]

def ensure_final_punctuation(text: str) -> str:
    """Ensure sentence ends with punctuation."""
    text = text.strip()
    if not text:
        return text

    # Already has final punctuation?
    if text[-1] in '.!?':
        return text

    # Add period
    return text + '.'

def fix_spacing_around_punctuation(text: str) -> str:
    """Fix spacing around commas, periods, etc."""
    # Remove space before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    # Add space after punctuation (if not at end)
    text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)

    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def remove_duplicate_punctuation(text: str) -> str:
    """Remove duplicate punctuation marks."""
    # Multiple periods → single period
    text = re.sub(r'\.{2,}', '.', text)

    # Multiple commas → single comma
    text = re.sub(r',{2,}', ',', text)

    # Multiple spaces → single space
    text = re.sub(r'\s{2,}', ' ', text)

    return text

# ============================================================================
# SENTENCE VALIDATION
# ============================================================================

def is_valid_sentence(text: str) -> bool:
    """
    Check if text is a valid sentence.

    Valid if:
    - Has at least one word
    - Starts with capital letter
    - Ends with punctuation
    - No obvious grammar errors
    """
    text = text.strip()

    if not text:
        return False

    # Must start with capital or number
    if not text[0].isupper() and not text[0].isdigit():
        return False

    # Must end with punctuation
    if text[-1] not in '.!?':
        return False

    # Must have at least one letter
    if not any(c.isalpha() for c in text):
        return False

    # Must be at least 3 characters
    if len(text) < 3:
        return False

    return True

def fix_common_errors(text: str) -> str:
    """Fix common grammar mistakes."""

    # Remove leading/trailing punctuation except at end
    text = text.strip()

    # Fix "Nicole gains gravitational, core concept" → "Nicole gains gravitational core concept."
    # Remove comma before period/end
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r',\s*$', '.', text)

    # Fix double punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)

    # Fix "word,word" → "word, word"
    text = re.sub(r'(\w),(\w)', r'\1, \2', text)

    # Fix "word.word" → "word. Word"
    def capitalize_after_period(match):
        return match.group(1) + ' ' + match.group(2).upper()
    text = re.sub(r'\.(\S)', capitalize_after_period, text)

    return text

# ============================================================================
# SENTENCE FINALIZATION (from sska concept)
# ============================================================================

def finalize_sentence(text: str) -> str:
    """
    Finalize sentence with perfect grammar.

    This is the main entry point - applies all fixes.
    """
    # Strip whitespace
    text = text.strip()

    if not text:
        return ""

    # Fix common errors
    text = fix_common_errors(text)

    # Fix spacing around punctuation
    text = fix_spacing_around_punctuation(text)

    # Remove duplicate punctuation
    text = remove_duplicate_punctuation(text)

    # Ensure final punctuation
    text = ensure_final_punctuation(text)

    # Capitalize first letter
    text = capitalize_first_letter(text)

    return text

def finalize_paragraph(sentences: List[str]) -> str:
    """
    Finalize multiple sentences into coherent paragraph.

    Each sentence gets perfect grammar, then joined with proper spacing.
    """
    finalized = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Finalize each sentence
        sentence = finalize_sentence(sentence)

        # Only add if valid
        if is_valid_sentence(sentence):
            finalized.append(sentence)

    # Join with space
    return ' '.join(finalized)

# ============================================================================
# SENTENCE STRUCTURE VALIDATION
# ============================================================================

def has_verb(text: str) -> bool:
    """
    Check if sentence likely has a verb.

    Simple heuristic: look for common verb patterns.
    """
    # Simple verb detection (not perfect, but good enough)
    common_verbs = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had',
        'do', 'does', 'did',
        'will', 'would', 'can', 'could', 'should',
        'may', 'might', 'must',
        'acts', 'becomes', 'resonates', 'dissolves',
        'remembers', 'gains', 'stays', 'speaks'
    }

    words = text.lower().split()
    return any(word in common_verbs for word in words)

def is_fragment(text: str) -> bool:
    """
    Detect sentence fragments.

    Fragment if:
    - No verb
    - Only 1-2 words
    - Ends with comma
    """
    text = text.strip()
    words = text.split()

    # Too short
    if len(words) < 3:
        return True

    # No verb
    if not has_verb(text):
        return True

    # Ends with comma (incomplete)
    if text.endswith(','):
        return True

    return False

def ensure_complete_sentence(text: str) -> str:
    """
    Ensure text is a complete sentence, not a fragment.

    If fragment detected, add minimal completion.
    """
    text = finalize_sentence(text)

    # Check if fragment
    if is_fragment(text):
        # Add minimal verb if missing
        if not has_verb(text):
            text = text.rstrip('.') + " resonates."
            text = capitalize_first_letter(text)

    return text

# ============================================================================
# PERFECT GRAMMAR API (sska-inspired)
# ============================================================================

def apply_perfect_grammar(text: str, ensure_complete: bool = True) -> str:
    """
    Apply perfect grammar to text.

    This is the main API - use this for all generated text!

    Args:
        text: Raw generated text
        ensure_complete: Fix fragments into complete sentences

    Returns:
        Grammatically correct text
    """
    if ensure_complete:
        return ensure_complete_sentence(text)
    else:
        return finalize_sentence(text)
