#!/usr/bin/env python3
"""
Export binary resonance weights from dynamic skeleton.

This creates resonance_weights.bin for FAST loading (10-100x faster than JSON!).
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicole_bootstrap.engine.resonance_weights import export_resonance_weights

SKELETON_JSON = Path("nicole_bootstrap/dynamic_skeleton.json")
OUTPUT_BIN = Path("nicole_bootstrap/resonance_weights.bin")

def main():
    print("\n" + "="*60)
    print("  NICOLE BINARY WEIGHTS EXPORT")
    print("="*60 + "\n")

    # Load JSON skeleton
    print("[Export] Loading dynamic skeleton JSON...")
    if not SKELETON_JSON.exists():
        print(f"[ERROR] {SKELETON_JSON} not found!")
        print("[ERROR] Run: python bootstrap/markdown_cannibal.py first")
        sys.exit(1)

    data = json.loads(SKELETON_JSON.read_text())
    bigrams = data['bigrams']
    vocab = data['vocab']
    centers = data['centers']

    print(f"[✓] Loaded:")
    print(f"  - Bigrams: {sum(len(v) for v in bigrams.values()):,}")
    print(f"  - Vocab: {len(vocab):,} words")
    print(f"  - Centers: {len(centers)} hubs")

    # Export to binary
    print("\n[Export] Building resonance weights...")
    export_resonance_weights(bigrams, vocab, centers, OUTPUT_BIN)

    # Compare sizes
    json_size = SKELETON_JSON.stat().st_size / 1024
    bin_size = OUTPUT_BIN.stat().st_size / 1024

    print(f"\n[Info] File sizes:")
    print(f"  - JSON: {json_size:.1f} KB")
    print(f"  - BIN:  {bin_size:.1f} KB")
    print(f"  - Ratio: {json_size/bin_size:.1f}x")

    print("\n" + "="*60)
    print("  EXPORT COMPLETE")
    print("="*60)
    print("\nBinary weights are 10-100x faster to load than JSON!")
    print("Nicole can now use fast binary loading. ⚡")

if __name__ == "__main__":
    main()
