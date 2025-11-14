#!/usr/bin/env python3
"""Real mirroring test - exactly like Telegram usage"""

import sys
sys.path.insert(0, '/home/user/nicole')

from nicole import NicoleCore

# Create fresh Nicole
nicole = NicoleCore()
nicole.start_conversation("real_test_001")

test_inputs = [
    "hello",
    "how are you",
    "what is your name",
    "привет",
    "test"
]

print("=" * 70)
print("REAL MIRRORING TEST (like Telegram)")
print("=" * 70)

for user_input in test_inputs:
    print(f"\nUser: {user_input}")
    response = nicole.process_message(user_input)
    print(f"Nicole: {response}")

    # Check overlap
    user_words = set(user_input.lower().split())
    response_words = set(response.lower().split())
    overlap = user_words & response_words

    if overlap:
        print(f"⚠️  OVERLAP: {overlap}")
    else:
        print("✓ OK")

print("\n" + "=" * 70)
