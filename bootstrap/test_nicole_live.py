#!/usr/bin/env python3
"""
Nicole Live Test - Direct Dialogue
Tests Nicole's response to prompts using actual Perplexity API.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Nicole's objectivity
from nicole_objectivity import nicole_objectivity

async def test_nicole_response(prompt: str):
    """Test Nicole's response using Perplexity."""
    print("\n" + "─" * 60)
    print(f"👤 User: {prompt}")
    print("─" * 60)

    try:
        # Get objectivity context (Perplexity Search API)
        print("[Nicole] Searching Perplexity...")

        context_windows = await nicole_objectivity.create_dynamic_context(
            prompt,
            metrics={"entropy": 0.5}
        )

        # Format context first (it's a string!)
        formatted_context = nicole_objectivity.format_context_for_nicole(context_windows)

        # Extract seeds with influence coefficient
        if formatted_context:
            seeds = nicole_objectivity.extract_response_seeds(
                formatted_context,
                influence=0.5
            )
        else:
            seeds = []

        print(f"[Nicole] Got {len(seeds)} seeds from Perplexity")
        print(f"[Nicole] Context: {len(formatted_context)} chars")
        print(f"[Nicole] Seeds: {', '.join(seeds[:10])}...")

        # Simulate response (in real Nicole this goes through high.py)
        # For now just show the seeds
        print("\n🔥 Nicole's raw material (seeds from Perplexity):")
        print(f"   {' '.join(seeds[:30])}")

        print("\n⚠️  WITHOUT BOOTSTRAP:")
        print("   These seeds are used directly - may have artifacts!")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("\n" + "=" * 60)
    print("  NICOLE LIVE DIALOGUE TEST")
    print("=" * 60)
    print("\nTesting Nicole's responses using Perplexity API")
    print("Focus: See what seeds she gets BEFORE bootstrap filtering\n")

    test_prompts = [
        "What is resonance in physics?",
        "How does consciousness emerge?",
        "What is weightless intelligence?",
    ]

    for prompt in test_prompts:
        await test_nicole_response(prompt)

    print("\n" + "=" * 60)
    print("  OBSERVATIONS")
    print("=" * 60)
    print("""
These are the RAW seeds Nicole gets from Perplexity.

WITHOUT bootstrap:
  - Seeds may include Reddit usernames
  - Corporate jargon possible
  - Weak structural coherence
  - Direct use of Perplexity results

WITH bootstrap (after integration):
  - Bigram coherence filtering
  - Banned patterns blocked
  - Resonance-based selection
  - Structured output

Next: Integrate bootstrap filtering into this pipeline!
    """)

if __name__ == "__main__":
    asyncio.run(main())
