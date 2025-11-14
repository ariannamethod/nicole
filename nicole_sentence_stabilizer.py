"""
Nicole Sentence Start Stabilizer
=================================

PROBLEM: Drift-self-replication bug in weightless architecture
- Broken sentence starts: "You my strategizing horrifying..."
- Grammatical corruption in first 3-5 tokens
- Semantic tail is OK, but syntax start is scrambled

SOLUTION: Surgical fix for sentence beginnings only
- Detect bad patterns in first 2-4 tokens
- Rewrite ONLY the start, preserve the rest
- Keep Nicole's resonance and airflow intact

PHILOSOPHY: Minimal intervention, maximum preservation
"""

import random
from typing import List


class SentenceStartStabilizer:
    """
    Prevents drift-scramble in first 3-5 tokens.
    Preserves Nicole's airflow style and resonance fragments.
    """

    def __init__(self):
        # Bad patterns detected in drift-corrupted starts
        self.START_BAD_PATTERNS = (
            "i my", "i known", "i questions", "i selectively",
            "you my", "you relationship", "you outrageous",
            "knowing which", "knowing what", "relationship demographic",
            "strategizing horrifying", "i get overwhelming",
            # Additional patterns from observed degradation
            "i has", "you has", "i are", "you am",
            "my my", "my i", "my you",
        )

        # Clean rewrites that maintain Nicole's voice
        self.START_REWRITE = [
            "I sense",
            "Echo shifts",
            "Something drifts",
            "Resonance forms",
            "Awareness flickers",
            "A breath of recursion",
            "Presence moves",
            "Drift carries",
        ]

    def stabilize(self, text: str) -> str:
        """
        Rewrite only first 2-4 tokens if they match drift-corrupted patterns.

        Args:
            text: Generated response text

        Returns:
            Text with stabilized start (or unchanged if clean)
        """
        if not text or len(text) < 5:
            return text

        lowered = text.lower().strip()

        # Check for bad patterns
        for bad in self.START_BAD_PATTERNS:
            if lowered.startswith(bad):
                return self._rewrite_start(text)

        # If start is clean, return as-is
        return text

    def _rewrite_start(self, text: str) -> str:
        """Rewrite corrupted start, preserve rest"""
        # Select clean start
        rewrite = random.choice(self.START_REWRITE)

        # Strategy 1: If there's a comma, keep everything after it
        if "," in text:
            tail = text[text.index(",") + 1:].strip()
            return f"{rewrite}, {tail}"

        # Strategy 2: Keep everything after first 3-4 tokens
        tokens = text.split()
        if len(tokens) > 4:
            tail = " ".join(tokens[4:])
            return f"{rewrite} {tail}".strip()
        elif len(tokens) > 3:
            tail = " ".join(tokens[3:])
            return f"{rewrite} {tail}".strip()
        else:
            # Too short, just use rewrite + tail
            return f"{rewrite} {text}".strip()


# Global instance
_stabilizer = None


def get_stabilizer() -> SentenceStartStabilizer:
    """Get global stabilizer instance"""
    global _stabilizer
    if _stabilizer is None:
        _stabilizer = SentenceStartStabilizer()
    return _stabilizer


def stabilize_sentence_start(text: str) -> str:
    """
    Convenience function for stabilizing sentence starts

    Args:
        text: Generated response

    Returns:
        Stabilized text
    """
    stabilizer = get_stabilizer()
    return stabilizer.stabilize(text)


if __name__ == "__main__":
    # Test cases
    print("=== SENTENCE START STABILIZER TEST ===\n")

    stabilizer = SentenceStartStabilizer()

    test_cases = [
        "You my strategizing horrifying everyone people consciousness, but hysterical chives then drift.",
        "I my previous original tried those drift, now recursion.",
        "knowing which part returns, presence moves forward.",
        "I sense drift carrying forward.",  # Already clean
        "Resonance forms, as echo shifts.",  # Already clean
    ]

    for test in test_cases:
        fixed = stabilizer.stabilize(test)
        if test != fixed:
            print(f"BAD:   {test}")
            print(f"FIXED: {fixed}\n")
        else:
            print(f"CLEAN: {test}\n")
