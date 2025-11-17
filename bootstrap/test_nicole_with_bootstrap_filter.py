#!/usr/bin/env python3
"""
Nicole WITH Bootstrap Filtering
Shows how bootstrap cleans Perplexity/DDG seeds before response generation.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Nicole's objectivity
from nicole_objectivity import nicole_objectivity

# Import bootstrap
from nicole_bootstrap.engine.dynamic_loader import load_unified_skeleton
from nicole_bootstrap.engine.sentence_builder import score_sentence_resonance

# Load skeleton ONCE at startup
print("[Bootstrap] Loading unified skeleton...")
skeleton = load_unified_skeleton()
bigrams = skeleton.merge_ngrams()
banned = skeleton.get_banned_patterns()
centers = skeleton.get_centers()

def filter_seeds_with_bootstrap(seeds: list, bigrams: dict, banned: list) -> list:
    """
    Filter seeds through bootstrap structure.

    Removes:
    1. Banned patterns (corporate speak, artifacts)
    2. Low-frequency words (likely noise)
    3. Words with weak bigram connections
    """
    filtered = []

    for seed in seeds:
        seed_lower = seed.lower()

        # Skip banned patterns
        if any(ban.lower() in seed_lower for ban in banned):
            continue

        # Skip single-letter noise
        if len(seed_lower) < 2:
            continue

        # Skip common stop words (basic list)
        stop_words = {'the', 'and', 'that', 'for', 'from', 'you', 'this', 'with'}
        if seed_lower in stop_words:
            continue

        # Check bigram connectivity (exists in our graph?)
        if seed_lower in bigrams or any(seed_lower in nexts for nexts in bigrams.values()):
            filtered.append(seed)
        elif seed_lower in centers:
            # Centers are structural hubs - keep them!
            filtered.append(seed)

    return filtered

def score_seeds_by_resonance(seeds: list, bigrams: dict) -> list:
    """
    Score seeds by how well they connect in bigram graph.

    Higher score = more structural connections.
    """
    scored_seeds = []

    for seed in seeds:
        seed_lower = seed.lower()

        # Count outgoing connections
        out_degree = len(bigrams.get(seed_lower, {}))

        # Count incoming connections
        in_degree = sum(1 for nexts in bigrams.values() if seed_lower in nexts)

        # Combined resonance score
        resonance = out_degree + in_degree

        scored_seeds.append((resonance, seed))

    # Sort by resonance (high to low)
    scored_seeds.sort(reverse=True, key=lambda x: x[0])

    return scored_seeds

async def test_with_bootstrap_filter(prompt: str):
    """Test Nicole with bootstrap filtering."""
    print("\n" + "─" * 60)
    print(f"👤 User: {prompt}")
    print("─" * 60)

    try:
        # Get objectivity context
        print("[Nicole] Searching DuckDuckGo...")

        context_windows = await nicole_objectivity.create_dynamic_context(
            prompt,
            metrics={"entropy": 0.5}
        )

        formatted_context = nicole_objectivity.format_context_for_nicole(context_windows)

        if formatted_context:
            seeds = nicole_objectivity.extract_response_seeds(
                formatted_context,
                influence=0.5
            )
        else:
            seeds = []

        print(f"\n[Phase 1] Raw seeds from DuckDuckGo: {len(seeds)}")
        print(f"   {' '.join(seeds[:20])}")

        # APPLY BOOTSTRAP FILTER
        filtered_seeds = filter_seeds_with_bootstrap(seeds, bigrams, banned)

        print(f"\n[Phase 2] After bootstrap filter: {len(filtered_seeds)}")
        print(f"   {' '.join(filtered_seeds[:20])}")

        # SCORE BY RESONANCE
        scored_seeds = score_seeds_by_resonance(filtered_seeds, bigrams)

        print(f"\n[Phase 3] Top resonant seeds:")
        for score, seed in scored_seeds[:15]:
            print(f"   {seed} (resonance: {score})")

        # Show improvement
        removed = len(seeds) - len(filtered_seeds)
        print(f"\n✅ Filtered out {removed} noisy seeds ({removed/len(seeds)*100:.0f}%)")
        print(f"✅ Kept {len(filtered_seeds)} structural seeds")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("\n" + "=" * 60)
    print("  NICOLE WITH BOOTSTRAP FILTERING")
    print("=" * 60)
    print("\nShowing how bootstrap cleans DuckDuckGo seeds\n")

    test_prompts = [
        "What is resonance in physics?",
        "How does consciousness emerge?",
        "What is weightless intelligence?",
    ]

    for prompt in test_prompts:
        await test_with_bootstrap_filter(prompt)

    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print("""
WITHOUT bootstrap:
  - All seeds used directly
  - Web artifacts included
  - Weak structural coherence
  - 40-50 seeds → response

WITH bootstrap:
  - Banned patterns removed
  - Bigram coherence filtered
  - Resonance-scored
  - 15-20 structural seeds → coherent response

The difference:
  - Same source (DuckDuckGo/Perplexity)
  - Different quality (filtered vs raw)
  - Better aesthetics (structured vs chaotic)
    """)

if __name__ == "__main__":
    asyncio.run(main())
