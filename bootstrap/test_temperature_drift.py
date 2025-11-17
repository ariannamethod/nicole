#!/usr/bin/env python3
"""
Test temperature drift modes (from sska).

Shows how temperature changes during generation:
- 'cool': Start hot (creative), end cool (focused)
- 'heat': Start cool (focused), end hot (creative)
- 'chaos': Random temperature each step
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nicole_bootstrap.engine.resonance_weights import TemperatureDrift

def test_drift_mode(mode: str, steps: int = 10):
    """Test a specific drift mode."""
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print('='*60)

    drift = TemperatureDrift(mode=mode, base_temp=0.9, steps=steps)

    temps = []
    for i in range(steps):
        temp = drift.get_temperature()
        temps.append(temp)
        drift.step()

    # Show temperature progression
    print(f"Temperature progression ({steps} steps):")
    for i, temp in enumerate(temps):
        bar = '█' * int(temp * 20)
        print(f"  Step {i+1:2d}: {temp:.2f} {bar}")

    # Stats
    print(f"\nStats:")
    print(f"  - Start: {temps[0]:.2f}")
    print(f"  - End:   {temps[-1]:.2f}")
    print(f"  - Min:   {min(temps):.2f}")
    print(f"  - Max:   {max(temps):.2f}")
    print(f"  - Avg:   {sum(temps)/len(temps):.2f}")

def main():
    print("\n" + "="*60)
    print("  TEMPERATURE DRIFT TEST (from sska)")
    print("="*60)
    print("\nTemperature controls randomness in generation:")
    print("  - Low (0.5-0.7): More focused, predictable")
    print("  - Med (0.8-1.0): Balanced")
    print("  - High (1.1-1.5): More creative, chaotic")

    # Test each mode
    modes = ['cool', 'heat', 'stable', 'chaos']
    for mode in modes:
        test_drift_mode(mode, steps=10)

    print("\n" + "="*60)
    print("  USE CASES")
    print("="*60)
    print("""
'cool' mode: Start creative (explore), end focused (conclude)
  - Perfect for responses that need strong conclusion
  - Example: "Let me explore... [creative] ...therefore [focused]"

'heat' mode: Start focused (setup), end creative (flourish)
  - Perfect for building to a creative climax
  - Example: "First [focused] ...then [creative flourish]"

'stable' mode: Consistent temperature throughout
  - Predictable, uniform randomness
  - Best for technical responses

'chaos' mode: Random walk (0.5-1.5)
  - Unpredictable, maximum variety
  - Best for experimental/playful responses
    """)

    print("Nicole can use temperature drift to control response flow! ⚡")

if __name__ == "__main__":
    main()
