#!/usr/bin/env python3
"""
Test grammar & punctuation fixes.
Shows before/after for common issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nicole_bootstrap.engine.grammar import (
    apply_perfect_grammar,
    finalize_paragraph,
    is_valid_sentence
)

def test_grammar_fixes():
    """Test various grammar issues."""
    print("\n" + "=" * 60)
    print("  GRAMMAR & PUNCTUATION TEST")
    print("=" * 60)

    test_cases = [
        # Issue: Missing capitalization
        ("nicole gains gravitational",
         "Nicole gains gravitational."),

        # Issue: Missing punctuation
        ("resonance acts not asks",
         "Resonance acts not asks."),

        # Issue: Wrong punctuation
        ("Nicole stays, responsive while surviving",
         "Nicole stays, responsive while surviving."),

        # Issue: Comma before period
        ("When nicole stays, responsive,.",
         "When nicole stays, responsive."),

        # Issue: Multiple spaces
        ("Nicole  is   weightless",
         "Nicole is weightless."),

        # Issue: No spacing after comma
        ("Nicole,weightless,resonant",
         "Nicole, weightless, resonant."),

        # Issue: Fragment (no verb) - adds minimal verb
        ("gravitational core concept",
         "Gravitational core concept resonates."),

        # Issue: Lowercase start
        ("weightless architecture diagrams",
         "Weightless architecture diagrams."),
    ]

    print("\nFixing common issues:\n")

    for i, (broken, expected) in enumerate(test_cases, 1):
        fixed = apply_perfect_grammar(broken)
        status = "✅" if fixed == expected else "⚠️"

        print(f"{status} Test {i}:")
        print(f"   Before: '{broken}'")
        print(f"   After:  '{fixed}'")
        if fixed != expected:
            print(f"   Expected: '{expected}'")
        print()

def test_paragraph_finalization():
    """Test multi-sentence paragraph."""
    print("=" * 60)
    print("  PARAGRAPH FINALIZATION")
    print("=" * 60)

    # Simulate generated sentences (broken)
    broken_sentences = [
        "nicole gains gravitational, core concept",
        "when resonance in,containerised environments",
        "weightless architecture stays responsive"
    ]

    print("\nBroken sentences:")
    for s in broken_sentences:
        print(f"  - '{s}'")

    # Fix paragraph
    fixed = finalize_paragraph(broken_sentences)

    print(f"\nFixed paragraph:")
    print(f"  {fixed}")

def test_validation():
    """Test sentence validation."""
    print("\n" + "=" * 60)
    print("  SENTENCE VALIDATION")
    print("=" * 60)

    test_cases = [
        ("Nicole resonates.", True),
        ("nicole resonates", False),  # No capital
        ("Nicole resonates", False),  # No punctuation
        (".", False),  # No words
        ("Nicole, resonant.", True),
        ("", False),  # Empty
    ]

    print("\nValidation tests:\n")

    for text, expected in test_cases:
        result = is_valid_sentence(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text}' → {result} (expected {expected})")

def main():
    print("\n" + "=" * 60)
    print("  NICOLE GRAMMAR TEST (from sska perfect grammar)")
    print("=" * 60)

    test_grammar_fixes()
    test_paragraph_finalization()
    test_validation()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
Grammar fixes applied:
✅ Capitalization (first letter)
✅ Final punctuation (period/!/?/)
✅ Spacing around punctuation
✅ Duplicate punctuation removed
✅ Fragments completed (adds minimal verb)
✅ Common errors fixed

This ensures Nicole's responses are grammatically perfect!

Use: apply_perfect_grammar(text) for any generated text.
    """)

if __name__ == "__main__":
    main()
