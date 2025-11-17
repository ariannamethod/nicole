"""
Nicole Bootstrap — Sentence Builder
Generates sentences using bigram walking + templates

Inspired by:
- sorokin.py: Template-based generation with resonance scoring
- sska/subjectivity.py: Bigram walking with temperature

This builder:
1. Loads dynamic skeleton (from markdown cannibal)
2. Uses templates for structure
3. Walks bigrams for word selection
4. Scores candidates by resonance
"""

from __future__ import annotations

import random
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# ============================================================================
# TEMPLATES (adapted from sorokin)
# ============================================================================

NICOLE_SENTENCE_TEMPLATES = [
    # Short, punchy (Nicole style)
    "{word1} {word2}.",
    "{word1} {word2} {word3}.",
    "{word1}, {word2}.",

    # Medium flow
    "{word1} {word2} {word3} {word4}.",
    "{word1} {word2}, {word3} {word4}.",
    "{word1} {word2} {word3}, {word4} {word5}.",

    # Longer, more complex
    "{word1} {word2} {word3}, where {word4} {word5}.",
    "{word1} {word2} through {word3} {word4}.",
    "When {word1} {word2}, {word3} {word4} {word5}.",
    "{word1} {word2} {word3} until {word4} {word5}.",

    # Nicole-specific patterns (from her speech)
    "{word1} acts, not asks.",
    "{word1} is {word2}, not {word3}.",
    "{word1} remembers {word2}.",
    "{word1} dissolves into {word2}.",
    "{word1} resonates with {word2}.",
    "No {word1}. Only {word2}.",
    "{word1} without {word2}.",
]

# ============================================================================
# BIGRAM WALKER
# ============================================================================

class BigramWalker:
    """Walks bigram graph to generate word sequences."""

    def __init__(self, bigrams: Dict[str, Dict[str, int]], vocab: List[str], centers: List[str]):
        self.bigrams = bigrams
        self.vocab = vocab
        self.centers = centers

    def walk(self, start_word: Optional[str] = None, max_steps: int = 10,
             temperature: float = 0.9) -> List[str]:
        """
        Random walk through bigram graph.

        Args:
            start_word: Starting word (random center if None)
            max_steps: Maximum walk length
            temperature: Sampling temperature (higher = more random)

        Returns:
            List of words
        """
        if start_word is None:
            # Start from a random center (high out-degree words)
            if self.centers:
                start_word = random.choice(self.centers)
            else:
                start_word = random.choice(self.vocab)

        words = [start_word.lower()]
        current = start_word.lower()

        for _ in range(max_steps - 1):
            if current not in self.bigrams:
                break

            next_words = self.bigrams[current]
            if not next_words:
                break

            # Temperature-based sampling
            if temperature >= 1.0:
                # Pure random
                current = random.choice(list(next_words.keys()))
            else:
                # Weighted by frequency (with temperature)
                candidates = list(next_words.items())
                weights = [count ** (1.0 / temperature) for _, count in candidates]
                total = sum(weights)
                weights = [w / total for w in weights]

                current = random.choices([w for w, _ in candidates], weights=weights)[0]

            words.append(current)

        return words

    def fill_template(self, template: str, word_pool: List[str]) -> str:
        """
        Fill template with words.

        Uses bigram walking from word_pool to maintain coherence.
        """
        # Extract placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)

        if not placeholders:
            return template

        # Start walk from random word in pool
        if word_pool:
            start = random.choice(word_pool)
        else:
            start = random.choice(self.vocab) if self.vocab else "word"

        words = self.walk(start, max_steps=len(placeholders), temperature=0.8)

        # Pad if needed
        while len(words) < len(placeholders):
            words.append(random.choice(self.vocab) if self.vocab else "word")

        # Fill template
        filled = template
        for i, placeholder in enumerate(placeholders):
            filled = filled.replace(f"{{{placeholder}}}", words[i], 1)

        return filled

# ============================================================================
# RESONANCE SCORING (from sorokin)
# ============================================================================

def count_syllables(word: str) -> int:
    """Rough syllable count."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return max(1, count)

def score_sentence_resonance(sentence: str, bigrams: Dict[str, Dict[str, int]]) -> float:
    """
    Score sentence by:
    1. Bigram coherence (how many bigrams exist in graph)
    2. Rhythmic variance (syllable patterns)
    3. Length preference (Nicole likes short/medium)
    """
    words = sentence.split()

    if len(words) < 2:
        return 0.0

    # 1. Bigram coherence
    valid_bigrams = 0
    for i in range(len(words) - 1):
        w1 = words[i].lower().strip('.,!?;:—-')
        w2 = words[i + 1].lower().strip('.,!?;:—-')

        if w1 in bigrams and w2 in bigrams.get(w1, {}):
            valid_bigrams += 1

    bigram_score = valid_bigrams / (len(words) - 1) if len(words) > 1 else 0.0

    # 2. Rhythmic variance
    syllables = [count_syllables(w.strip('.,!?;:—-')) for w in words]
    if len(syllables) > 1:
        mean_syl = sum(syllables) / len(syllables)
        variance = sum((s - mean_syl) ** 2 for s in syllables) / len(syllables)
        rhythm_score = min(variance / 2.0, 1.0)
    else:
        rhythm_score = 0.0

    # 3. Length preference (Nicole likes 5-15 words)
    word_count = len(words)
    if 5 <= word_count <= 15:
        length_score = 1.0
    elif word_count < 5:
        length_score = word_count / 5.0
    else:
        length_score = max(0.0, 1.0 - (word_count - 15) / 10.0)

    # Combined score
    return bigram_score * 0.5 + rhythm_score * 0.3 + length_score * 0.2

# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_nicole_sentence(
    bigrams: Dict[str, Dict[str, int]],
    vocab: List[str],
    centers: List[str],
    seed_words: Optional[List[str]] = None,
    temperature: float = 0.9,
    n_candidates: int = 10
) -> str:
    """
    Generate a single Nicole-style sentence.

    Process:
    1. Select random template
    2. Generate multiple candidates
    3. Score by resonance
    4. Return best

    Args:
        bigrams: Bigram graph
        vocab: Vocabulary
        centers: High out-degree words
        seed_words: Optional seed words to start from
        temperature: Sampling temperature
        n_candidates: Number of candidates to generate

    Returns:
        Generated sentence
    """
    walker = BigramWalker(bigrams, vocab, centers)

    # Generate candidates
    candidates = []

    for _ in range(n_candidates):
        # Pick random template
        template = random.choice(NICOLE_SENTENCE_TEMPLATES)

        # Fill template
        if seed_words:
            sentence = walker.fill_template(template, seed_words)
        else:
            # Use centers as seed
            sentence = walker.fill_template(template, centers[:20])

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        # Score
        score = score_sentence_resonance(sentence, bigrams)
        candidates.append((score, sentence))

    # Return best
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]

def generate_nicole_paragraph(
    bigrams: Dict[str, Dict[str, int]],
    vocab: List[str],
    centers: List[str],
    seed_words: Optional[List[str]] = None,
    n_sentences: int = 3,
    temperature: float = 0.9
) -> str:
    """
    Generate multi-sentence paragraph.

    Each sentence is independently generated and scored.
    """
    sentences = []

    for _ in range(n_sentences):
        sentence = generate_nicole_sentence(
            bigrams, vocab, centers,
            seed_words=seed_words,
            temperature=temperature,
            n_candidates=10
        )
        sentences.append(sentence)

    return ' '.join(sentences)

# ============================================================================
# CONVENIENCE API
# ============================================================================

def load_dynamic_skeleton(path: Path) -> Tuple[Dict, List, List]:
    """
    Load dynamic skeleton JSON.

    Returns:
        (bigrams, vocab, centers)
    """
    data = json.loads(path.read_text())
    return data['bigrams'], data['vocab'], data['centers']
