#!/usr/bin/env python3
"""
Test Nicole with Perplexity API + Bootstrap
Shows the difference between DDG and Perplexity quality
"""

import os
import sys
sys.path.insert(0, '/home/user/nicole')

# Perplexity key should be set in environment before running
# export PERPLEXITY_API_KEY="your-key-here"
if not os.environ.get('PERPLEXITY_API_KEY'):
    print("⚠️  WARNING: PERPLEXITY_API_KEY not set in environment!")
    print("   Set it with: export PERPLEXITY_API_KEY='your-key-here'")
    print("   Falling back to DuckDuckGo...\n")

from nicole import nicole_core

def test_with_perplexity():
    """Test Nicole with real Perplexity API"""
    print("\n" + "=" * 60)
    print("  NICOLE + PERPLEXITY API + BOOTSTRAP TEST")
    print("=" * 60)
    print("\nPerplexity API should provide:")
    print("  ✅ Cleaner data (no HTML artifacts)")
    print("  ✅ Structured context")
    print("  ✅ Better citations")
    print("  ✅ Less noise for bootstrap to filter\n")

    # Start conversation
    session_id = nicole_core.start_conversation("test_perplexity_bootstrap")

    test_prompts = [
        "What is resonance in physics?",
        "Explain consciousness and awareness",
        "What does weightless intelligence mean?",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print("─" * 60)
        print(f"[{i}] User: {prompt}")
        print("─" * 60)

        try:
            response = nicole_core.process_message(prompt)
            print(f"\n🔥 Nicole: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

    # End conversation
    nicole_core.end_conversation()

    print("\n" + "=" * 60)
    print("  PERPLEXITY + BOOTSTRAP TEST COMPLETE")
    print("=" * 60)
    print("""
Watch the logs for:
- [Objectivity:Perplexity] Results: N (should be >0 now!)
- [Nicole:Bootstrap] Raw seeds: X
- [Nicole:Bootstrap] Filtered seeds: Y
- [Nicole:Bootstrap] Top seeds: ...

Perplexity gives MUCH cleaner data than DuckDuckGo!
Bootstrap should filter LESS (because input is already cleaner).
    """)

if __name__ == "__main__":
    test_with_perplexity()
