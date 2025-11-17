"""
Nicole Bootstrap — Sentence Planner
Chooses structural plan for sentence generation

STUB: Will be integrated into Nicole's generation pipeline later
"""

import random
from . import loader

def choose_structure(prompt, world_context, self_state, drift_state):
    """
    Choose sentence structure based on context

    Returns:
        dict with:
        - shape: phrase template
        - style: punctuation/length preferences
        - ngrams: preferred bigrams/trigrams
        - banned: patterns to avoid
    """

    # Load skeleton
    shapes = loader.get_shapes()
    style = loader.get_style()
    ngrams = loader.get_ngrams()
    banned = loader.get_banned()

    # Select shape (weighted by count if available)
    if shapes:
        total_count = sum(s.get("count", 1) for s in shapes)
        weights = [s.get("count", 1) / total_count for s in shapes]
        chosen_shape = random.choices(shapes, weights=weights, k=1)[0]
    else:
        chosen_shape = {"pattern": "default", "length": 10, "count": 0}

    # Filter n-grams by prompt keywords
    prompt_tokens = set(prompt.lower().split())
    relevant_bigrams = [
        bg for bg in ngrams.get("bigrams", [])
        if any(token in prompt_tokens for token in bg.get("tokens", []))
    ][:10]  # Top 10 relevant

    return {
        "shape": chosen_shape,
        "style": style,
        "ngrams": {
            "bigrams": relevant_bigrams
        },
        "banned": banned
    }

def filter_banned(text, banned_patterns):
    """Check if text contains banned patterns"""
    text_lower = text.lower()
    for pattern in banned_patterns:
        if pattern.lower() in text_lower:
            return False  # Reject
    return True  # Accept
