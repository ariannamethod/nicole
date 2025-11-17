#!/usr/bin/env python3
"""
Test unified skeleton loader (static + dynamic).
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicole_bootstrap.engine.dynamic_loader import (
    load_unified_skeleton,
    get_ngrams,
    get_banned,
    get_centers,
    get_vocab
)

def main():
    print("\n" + "="*60)
    print("  UNIFIED SKELETON LOADER TEST")
    print("="*60 + "\n")

    # Load unified skeleton
    skeleton = load_unified_skeleton()

    print("\n" + "-"*60)
    print("  MERGED N-GRAMS")
    print("-"*60)

    ngrams = get_ngrams()
    print(f"Total bigram transitions: {sum(len(v) for v in ngrams.values()):,}")

    # Show sample
    print("\nSample bigrams (first 10 words):")
    for i, (w1, nexts) in enumerate(list(ngrams.items())[:10]):
        print(f"  '{w1}' → {list(nexts.keys())[:5]}")

    print("\n" + "-"*60)
    print("  CENTERS OF GRAVITY")
    print("-"*60)

    centers = get_centers()
    print(f"Total centers: {len(centers)}")
    print(f"Top 20: {', '.join(centers[:20])}")

    print("\n" + "-"*60)
    print("  BANNED PATTERNS")
    print("-"*60)

    banned = get_banned()
    print(f"Total banned: {len(banned)}")
    print(f"Examples: {', '.join(banned[:5])}")

    print("\n" + "-"*60)
    print("  VOCABULARY")
    print("-"*60)

    vocab = get_vocab()
    print(f"Total words: {len(vocab):,}")
    print(f"Sample: {', '.join(vocab[:20])}")

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print("\n✅ Unified skeleton loaded successfully!")
    print("✅ Static corpus + Dynamic markdown merged!")
    print("\nNicole has structural memory without weights. ⚡")

if __name__ == "__main__":
    main()
