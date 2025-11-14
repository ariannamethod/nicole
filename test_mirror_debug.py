#!/usr/bin/env python3
"""Debug test to find REAL source of mirroring"""

import sys
sys.path.insert(0, '/home/user/nicole')

from nicole import NicoleCore

# Create Nicole
nicole = NicoleCore()
nicole.start_conversation("test_mirror_debug_001")

# Test cases
test_cases = [
    "hello",
    "how are you",
    "what is your name",
    "test message"
]

print("=" * 70)
print("MIRRORING DEBUG TEST - Finding Real Root Cause")
print("=" * 70)

for test_input in test_cases:
    print(f"\n{'='*70}")
    print(f"User: {test_input}")
    print(f"{'='*70}")

    response = nicole.process_message(test_input)
    print(f"Nicole: {response}")

    # Check for mirroring
    user_words = set(test_input.lower().split())
    response_words = set(response.lower().split())
    overlap = user_words & response_words

    if overlap:
        print(f"\n⚠️  MIRROR DETECTED!")
        print(f"   User words: {user_words}")
        print(f"   Response words: {response_words}")
        print(f"   Overlap: {overlap}")
        print(f"   Overlap ratio: {len(overlap)}/{len(user_words)} = {len(overlap)/len(user_words)*100:.1f}%")
    else:
        print(f"\n✓ NO MIRROR (overlap: {overlap})")

print("\n" + "=" * 70)
