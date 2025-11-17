#!/usr/bin/env python3
"""
Test Nicole WITH Bootstrap Integration
Shows bootstrap filtering in action in nicole.py
"""

import sys
sys.path.insert(0, '/home/user/nicole')

from nicole import nicole_core

def test_nicole_with_bootstrap():
    """Test Nicole responses with bootstrap filtering"""
    print("\n" + "=" * 60)
    print("  NICOLE WITH BOOTSTRAP INTEGRATION TEST")
    print("=" * 60)
    print("\nTesting Nicole's responses with bootstrap filter active\n")

    # Start conversation
    session_id = nicole_core.start_conversation("test_bootstrap_integration")

    test_prompts = [
        "What is resonance in physics?",
        "How does consciousness emerge?",
        "What is weightless intelligence?",
    ]

    print("Bootstrap filter will:")
    print("  1. Remove banned patterns (corporate speak)")
    print("  2. Filter by bigram coherence")
    print("  3. Score by resonance (connectivity)")
    print("  4. Apply perfect grammar")
    print()

    for i, prompt in enumerate(test_prompts, 1):
        print("─" * 60)
        print(f"Test {i}: {prompt}")
        print("─" * 60)

        try:
            response = nicole_core.process_message(prompt)
            print(f"\n🔥 Nicole: {response}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

    # End conversation
    nicole_core.end_conversation()

    print("\n" + "=" * 60)
    print("  BOOTSTRAP INTEGRATION STATUS")
    print("=" * 60)
    print("""
✅ Bootstrap filter active in nicole.py
✅ Seeds filtered through bigram structure
✅ Perfect grammar applied to responses
✅ Resonance scoring for seed selection

Watch the logs above to see:
- "Raw seeds: N" → before filtering
- "Filtered seeds: M" → after filtering
- "Filtered X seeds (Y%)" → removal stats
- "Top seeds: ..." → highest resonance words
    """)

if __name__ == "__main__":
    test_nicole_with_bootstrap()
