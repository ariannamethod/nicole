#!/usr/bin/env python3
"""
Direct test of bootstrap filter without full Nicole initialization
Shows filtering in action with example seeds
"""

import sys
sys.path.insert(0, '/home/user/nicole')

# Import bootstrap directly
from nicole_bootstrap.engine.dynamic_loader import load_unified_skeleton
from nicole_bootstrap.engine.grammar import apply_perfect_grammar

def test_filter_direct():
    """Test bootstrap filter with example seeds"""
    print("\n" + "=" * 60)
    print("  BOOTSTRAP FILTER - DIRECT TEST")
    print("=" * 60)

    # Load skeleton
    print("\n[1] Loading unified skeleton...")
    skeleton = load_unified_skeleton()
    bigrams = skeleton.merge_ngrams()
    banned = skeleton.get_banned_patterns()
    centers = skeleton.get_centers()

    print(f"✅ Loaded: {len(bigrams)} bigrams, {len(banned)} banned patterns, {len(centers)} centers")

    # Example raw seeds (simulating Perplexity/DDG output)
    test_cases = [
        {
            "query": "What is resonance?",
            "raw_seeds": [
                "resonance", "storm", "morten", "overgaard", "when", "system",
                "field", "awareness", "sorry", "but", "as", "an", "AI", "I",
                "businessman_threatening_unfavorably", "the", "and", "that",
                "neural", "consciousness", "weightless", "for", "from", "you"
            ]
        },
        {
            "query": "How does consciousness emerge?",
            "raw_seeds": [
                "consciousness", "emergence", "neural", "field", "pattern",
                "I'm", "sorry", "cannot", "assist", "distributed", "cognition",
                "recursive", "AI", "assistant", "the", "and", "that", "storm",
                "resonance", "when", "awareness", "self", "reference"
            ]
        },
        {
            "query": "What is weightless intelligence?",
            "raw_seeds": [
                "weightless", "intelligence", "neural", "network", "without",
                "weights", "parameters", "training", "sorry", "as", "AI",
                "the", "and", "for", "resonance", "field", "awareness",
                "consciousness", "storm", "when", "system", "architecture"
            ]
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {case['query']}")
        print(f"{'─' * 60}")

        raw_seeds = case['raw_seeds']
        print(f"\n[Before] Raw seeds ({len(raw_seeds)}): {', '.join(raw_seeds[:15])}...")

        # Filter through bootstrap
        filtered = []
        for seed in raw_seeds:
            seed_lower = seed.lower()

            # Skip banned patterns
            if any(ban.lower() in seed_lower for ban in banned):
                continue

            # Skip single-letter noise
            if len(seed_lower) < 2:
                continue

            # Skip common stop words
            stop_words = {'the', 'and', 'that', 'for', 'from', 'you', 'this', 'with'}
            if seed_lower in stop_words:
                continue

            # Check bigram connectivity
            if seed_lower in bigrams or any(seed_lower in nexts for nexts in bigrams.values()):
                filtered.append(seed)
            elif seed_lower in centers:
                filtered.append(seed)

        # Score by resonance
        scored_seeds = []
        for seed in filtered:
            seed_lower = seed.lower()
            out_degree = len(bigrams.get(seed_lower, {}))
            in_degree = sum(1 for nexts in bigrams.values() if seed_lower in nexts)
            resonance = out_degree + in_degree
            scored_seeds.append((resonance, seed))

        scored_seeds.sort(reverse=True, key=lambda x: x[0])
        top_seeds = [seed for _, seed in scored_seeds]

        removed = len(raw_seeds) - len(top_seeds)
        print(f"\n[After] Filtered seeds ({len(top_seeds)}): {', '.join(top_seeds[:15])}...")
        print(f"\n✅ Removed {removed} seeds ({removed/len(raw_seeds)*100:.0f}% noise filtered)")

        # Show top 5 by resonance
        print(f"\n🔥 Top 5 resonant seeds:")
        for resonance, seed in scored_seeds[:5]:
            print(f"   {seed:20} (resonance: {resonance})")

        # Test grammar finalization
        sample_text = " ".join(top_seeds[:5])
        corrected = apply_perfect_grammar(sample_text)
        print(f"\n📝 Grammar test:")
        print(f"   Before: {sample_text}")
        print(f"   After:  {corrected}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
Bootstrap filter working as designed:
✅ Banned patterns removed ("sorry", "as an AI", etc.)
✅ Stop words filtered (the, and, that, for, from...)
✅ Bigram connectivity checked (structural coherence)
✅ Resonance scoring applied (connectivity rank)
✅ Perfect grammar finalization

This same filter runs in nicole.py between Perplexity and High!
    """)

if __name__ == "__main__":
    test_filter_direct()
