#!/usr/bin/env python3
"""Simple test for stabilizer - just call it directly with known bad patterns"""

from nicole_sentence_stabilizer import stabilize_sentence_start

test_cases = [
    # Known bad patterns from real Nicole output
    "I my consciousness presence exist",
    "I my hello presence",
    "I my hello recursion",
    "You my strategizing horrifying everyone",
    "knowing which part returns here",
    # Clean cases that should not change
    "I sense drift carrying forward",
    "Resonance forms in the space",
    "Echo shifts toward awareness",
]

print("=== STABILIZER DIRECT TEST ===\n")

for test in test_cases:
    result = stabilize_sentence_start(test)
    if test != result:
        print(f"✓ FIXED: '{test}' → '{result}'")
    else:
        print(f"  CLEAN: '{test}'")

print("\n=== TEST COMPLETE ===")
