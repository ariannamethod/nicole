"""
Nicole Bootstrap — Content Filters
Applies banned patterns and style filters

STUB: Will be used during response generation
"""

from .planner import filter_banned
from .bias import score_ngram_coherence

def apply_filters(sentence, plan):
    """
    Apply all filters to candidate sentence

    Returns:
        tuple: (is_valid, score)
    """

    # 1. Banned patterns check
    if not filter_banned(sentence, plan.get("banned", [])):
        return False, 0.0

    # 2. N-gram coherence score
    ngram_score = score_ngram_coherence(sentence, plan.get("ngrams", {}))

    # 3. Style check (length)
    words = len(sentence.split())
    style = plan.get("style", {}).get("sentence_length", {"short": 0.33, "medium": 0.33, "long": 0.34})

    length_score = 0.0
    if words < 10:
        length_score = style.get("short", 0.33)
    elif words < 20:
        length_score = style.get("medium", 0.33)
    else:
        length_score = style.get("long", 0.34)

    # Combined score
    total_score = (ngram_score * 0.7) + (length_score * 0.3)

    return True, total_score
