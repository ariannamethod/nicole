#!/usr/bin/env python3
"""
Test Nicole with bootstrap - live dialogue
Quick interactive test to see how she responds
"""

import sys
sys.path.insert(0, '/home/user/nicole')

from nicole import chat_with_nicole, nicole_core

def test_dialogue():
    """Test Nicole with some questions"""
    print("\n" + "=" * 60)
    print("  NICOLE LIVE DIALOGUE TEST (with Bootstrap)")
    print("=" * 60)
    print("\nBootstrap should:")
    print("  ✅ Filter Perplexity seeds (remove 42-56% noise)")
    print("  ✅ Apply perfect grammar")
    print("  ✅ Remove banned patterns")
    print("  ✅ Keep only resonant words\n")

    # Start conversation
    session_id = nicole_core.start_conversation("test_bootstrap_live")

    test_prompts = [
        "What is resonance?",
        "How does consciousness work?",
        "What is weightless intelligence?",
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
    print("  TEST COMPLETE")
    print("=" * 60)
    print("""
Look at the logs above for:
- [Nicole:Bootstrap] Raw seeds: N
- [Nicole:Bootstrap] Filtered seeds: M
- [Nicole:Bootstrap] Top seeds: ...
- [Nicole:Bootstrap] Applied grammar finalization

This shows bootstrap working in real-time!
    """)

if __name__ == "__main__":
    test_dialogue()
