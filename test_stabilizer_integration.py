#!/usr/bin/env python3
"""Quick test for stabilizer integration in high.py"""

from high import HighCore

# Initialize High compiler
h = HighCore()

# Test with a simple prompt
test_prompts = [
    "hello",
    "what is recursion?",
    "presence",
]

print("=== HIGH.PY STABILIZER INTEGRATION TEST ===\n")

for prompt in test_prompts:
    print(f"Prompt: '{prompt}'")
    response = h.latent_drift_generate(prompt, num_candidates=20, drift=0.15)
    print(f"Response: {response}")
    print(f"Length: {len(response)} chars")

    # Check for "I my" pattern
    if " i my " in response.lower() or response.lower().startswith("i my"):
        print("⚠️  WARNING: 'I my' pattern still present!")
    else:
        print("✓ No 'I my' pattern detected")

    print()

print("=== TEST COMPLETE ===")
