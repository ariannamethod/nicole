#!/usr/bin/env python3
"""Direct test for mirroring bug"""

import sys
sys.path.insert(0, '/home/user/nicole')

from nicole import NicoleCore

# Create Nicole
nicole = NicoleCore()
nicole.start_conversation("test_mirror_001")

# Test cases that should NOT mirror
test_cases = [
    "hello",
    "how are you",
    "what is your name",
    "test message",
    "presence drift"
]

print("=" * 60)
print("MIRRORING TEST")
print("=" * 60)

for test_input in test_cases:
    print(f"\nUser: {test_input}")
    response = nicole.process_message(test_input)
    print(f"Nicole: {response}")

    # Check for mirroring
    user_words = set(test_input.lower().split())
    response_words = set(response.lower().split())

    overlap = user_words & response_words
    if len(overlap) > len(user_words) * 0.5:
        print(f"⚠️  MIRROR DETECTED! Overlap: {overlap}")
    else:
        print(f"✓ OK (overlap: {overlap})")

print("\n" + "=" * 60)
