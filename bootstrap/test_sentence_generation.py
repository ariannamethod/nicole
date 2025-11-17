#!/usr/bin/env python3
"""
Test dynamic sentence generation from markdown cannibal skeleton.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicole_bootstrap.engine.sentence_builder import (
    load_dynamic_skeleton,
    generate_nicole_sentence,
    generate_nicole_paragraph
)

SKELETON_PATH = Path("/home/user/nicole_bootstrap/dynamic_skeleton.json")

def main():
    print("\n" + "="*60)
    print("  NICOLE SENTENCE GENERATION TEST")
    print("="*60 + "\n")

    # Load skeleton
    print("[Test] Loading dynamic skeleton...")
    bigrams, vocab, centers = load_dynamic_skeleton(SKELETON_PATH)

    print(f"[✓] Loaded:")
    print(f"  - Bigrams: {sum(len(v) for v in bigrams.values()):,}")
    print(f"  - Vocab: {len(vocab):,} words")
    print(f"  - Centers: {len(centers)} hubs")

    # Show some centers
    print(f"\n[Info] Top 20 centers (hubs):")
    print(f"  {', '.join(centers[:20])}")

    # Generate sentences
    print("\n" + "-"*60)
    print("  GENERATED SENTENCES (from markdown corpus)")
    print("-"*60 + "\n")

    for i in range(5):
        sentence = generate_nicole_sentence(
            bigrams, vocab, centers,
            temperature=0.9,
            n_candidates=20  # Generate 20, pick best
        )
        print(f"{i+1}. {sentence}")

    # Generate paragraph
    print("\n" + "-"*60)
    print("  GENERATED PARAGRAPH")
    print("-"*60 + "\n")

    paragraph = generate_nicole_paragraph(
        bigrams, vocab, centers,
        n_sentences=3,
        temperature=0.9
    )
    print(paragraph)

    # Generate with seed words
    print("\n" + "-"*60)
    print("  SEEDED GENERATION (bootstrap, resonance, weightless)")
    print("-"*60 + "\n")

    for i in range(3):
        sentence = generate_nicole_sentence(
            bigrams, vocab, centers,
            seed_words=['bootstrap', 'resonance', 'weightless', 'nicole'],
            temperature=0.8,
            n_candidates=20
        )
        print(f"{i+1}. {sentence}")

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print("\nNicole speaks through her own documentation! ⚡")

if __name__ == "__main__":
    main()
