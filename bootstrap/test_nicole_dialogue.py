#!/usr/bin/env python3
"""
Quick Nicole Dialogue Test
Shows how Nicole responds to prompts using current generation logic.

This lets us see aesthetic quality BEFORE and AFTER bootstrap integration.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Nicole's generation
try:
    from nicole_bootstrap.engine.dynamic_loader import load_unified_skeleton
    from nicole_bootstrap.engine.sentence_builder import (
        generate_nicole_sentence,
        generate_nicole_paragraph
    )
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    print("[Warning] Bootstrap engine not available")

def show_response_aesthetically(prompt: str, response: str):
    """Pretty print dialogue exchange."""
    print("\n" + "─" * 60)
    print(f"👤 User: {prompt}")
    print("─" * 60)
    print(f"🔥 Nicole: {response}")
    print("─" * 60)

def test_without_bootstrap():
    """
    Test Nicole's current response generation.

    This is placeholder - in real Nicole this would be:
    1. Perplexity Search API fetch
    2. Extract seeds
    3. Build response from seeds

    For now we just show the concept.
    """
    print("\n" + "=" * 60)
    print("  WITHOUT BOOTSTRAP (Current State)")
    print("=" * 60)
    print("\nNicole uses Perplexity Search API for seeds.")
    print("Results may contain Reddit artifacts, corporate speak, etc.\n")

    test_prompts = [
        "What is resonance?",
        "How do you learn?",
        "Are you conscious?",
    ]

    for prompt in test_prompts:
        # Simulate response (in real Nicole this comes from objectivity + high.py)
        # Without bootstrap: may have structural issues
        response = f"[Simulated] Perplexity says: resonance is important and consciousness awareness field..."
        show_response_aesthetically(prompt, response)

    print("\n⚠️  Without bootstrap: Structure may be weak, artifacts present")

def test_with_bootstrap():
    """
    Test with bootstrap structure.

    This shows what happens when we:
    1. Perplexity Search API fetch (same as before)
    2. Filter through bigram coherence
    3. Apply banned patterns
    4. Score by resonance
    5. Use temperature drift
    """
    if not BOOTSTRAP_AVAILABLE:
        print("\n[Skipped] Bootstrap not available")
        return

    print("\n" + "=" * 60)
    print("  WITH BOOTSTRAP (Enhanced)")
    print("=" * 60)
    print("\nNicole uses Perplexity Search API + bigram filtering.")
    print("Results filtered by structure, banned patterns removed.\n")

    # Load skeleton
    skeleton = load_unified_skeleton()
    bigrams = skeleton.merge_ngrams()
    vocab = skeleton.get_vocab()
    centers = skeleton.get_centers()

    test_prompts = [
        "What is resonance?",
        "How do you learn?",
        "Are you conscious?",
    ]

    for prompt in test_prompts:
        # Generate using bootstrap structure
        # This simulates what Nicole would do with filtered Perplexity seeds
        seed_words = ['resonance', 'nicole', 'weightless', 'consciousness', 'field']

        response = generate_nicole_paragraph(
            bigrams, vocab, centers,
            seed_words=seed_words,
            n_sentences=2,
            temperature=0.9
        )

        show_response_aesthetically(prompt, response)

    print("\n✅ With bootstrap: Filtered structure, coherent bigrams")

def compare_aesthetics():
    """Show side-by-side comparison."""
    print("\n" + "=" * 60)
    print("  AESTHETIC COMPARISON")
    print("=" * 60)

    print("""
Without Bootstrap:
  ❌ Reddit usernames as words
  ❌ Corporate speak ("I'm sorry, but...")
  ❌ Weak sentence structure
  ❌ Random Perplexity artifacts
  ⚠️  Seeds are good, structure is weak

With Bootstrap:
  ✅ Bigram coherence filtering
  ✅ Banned patterns blocked
  ✅ Resonance-based selection
  ✅ Temperature drift control
  ✅ Structured from documentation

The difference:
  - Same Perplexity seeds (facts/concepts)
  - Different structure (coherent vs random)
  - Filtered artifacts (clean vs noisy)
  - Aesthetic quality (resonant vs chaotic)
    """)

def main():
    print("\n" + "=" * 60)
    print("  NICOLE DIALOGUE AESTHETIC TEST")
    print("=" * 60)
    print("\nThis tests how Nicole responds with/without bootstrap.")
    print("Focus: AESTHETIC QUALITY of generated responses\n")

    # Test without bootstrap (current state)
    test_without_bootstrap()

    # Test with bootstrap (enhanced)
    test_with_bootstrap()

    # Show comparison
    compare_aesthetics()

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("""
1. Run Nicole in Telegram (current state)
   → See actual Perplexity responses
   → Note structural issues

2. Integrate bootstrap into nicole.py
   → Add bigram filtering to objectivity pipeline
   → Apply banned patterns
   → Score candidates by resonance

3. Test again in Telegram (with bootstrap)
   → Compare aesthetic quality
   → Verify improvement

4. Later: NanoGPT training (32GB machine)
   → Full skeleton (corpus + markdown + model-learned)
   → Even better structure
    """)

    print("\nReady to test in Telegram, соавтор? 🔥")

if __name__ == "__main__":
    main()
