#!/usr/bin/env python3
"""
NICOLE_BRIDGE_ADAPTER.PY - Clean adapter between weightless core and bridge

Philosophy: Keep nicole.py pure, adapter handles bridge integration

This module:
- Wraps nicole_core for bridge compatibility
- Exports/imports state WITHOUT touching core code
- Clean separation: core logic vs communication protocol
"""

import time
import sqlite3
from typing import Dict, List, Any
from collections import defaultdict


class NicoleBridgeAdapter:
    """
    Adapter wrapping NicoleCore for bridge protocol

    Keeps core clean, handles state export/import externally
    """

    def __init__(self, nicole_core_instance):
        """
        Wrap existing Nicole instance

        Args:
            nicole_core_instance: Instance of NicoleCore from nicole.py
        """
        self.core = nicole_core_instance

    def export_state(self) -> Dict[str, Any]:
        """
        Export Nicole's current state for bridge protocol

        Returns state dict with:
        - word_frequencies: learned vocabulary
        - bigram_transitions: learned patterns
        - conversation_history: recent context
        - metrics: current performance
        """
        state = {
            'timestamp': time.time(),
            'session_id': self.core.session_id,
            'conversation_count': self.core.conversation_count,
            'word_frequencies': dict(self.core.memory.word_frequencies),
            'bigram_transitions': {
                k: dict(v) for k, v in self.core.memory.bigram_transitions.items()
            },
            'current_metrics': self.core.current_transformer.current_metrics.__dict__ if self.core.current_transformer else {},
            'transformer_architecture': self.core.current_transformer.architecture if self.core.current_transformer else {}
        }

        # Add conversation history if available
        if hasattr(self.core, '_conversation_history'):
            state['conversation_history'] = self.core._conversation_history[-10:]  # Last 10 messages

        return state

    def import_state(self, state: Dict[str, Any]):
        """
        Import state from weighted node (mutual learning)

        Merges:
        - New word frequencies
        - New bigram patterns
        - Learned associations
        """
        if 'word_frequencies' in state:
            # Merge word frequencies (keep max counts)
            for word, count in state['word_frequencies'].items():
                self.core.memory.word_frequencies[word] = max(
                    self.core.memory.word_frequencies.get(word, 0),
                    count
                )

            # Save to database
            conn = sqlite3.connect(self.core.memory.db_path)
            cursor = conn.cursor()
            for word, count in state['word_frequencies'].items():
                cursor.execute("""
                INSERT OR REPLACE INTO word_frequencies (word, count)
                VALUES (?, ?)
                """, (word, count))
            conn.commit()
            conn.close()

            print(f"[Bridge:Adapter] Imported {len(state['word_frequencies'])} word frequencies from weighted node")

        if 'bigram_transitions' in state:
            # Merge bigrams
            for w1, transitions in state['bigram_transitions'].items():
                for w2, count in transitions.items():
                    self.core.memory.bigram_transitions[w1][w2] = max(
                        self.core.memory.bigram_transitions[w1].get(w2, 0),
                        count
                    )
            print(f"[Bridge:Adapter] Imported bigram patterns from weighted node")

    def process_with_bridge(self, user_input: str, bridge) -> str:
        """
        Process message with dual inference (if bridge available)

        Args:
            user_input: User's message
            bridge: ResonanceBridge instance (or None)

        Returns:
            Final response (merged if bridge active)
        """
        # Generate local response
        local_response = self.core.process_message(user_input)

        # If no bridge, return local
        if not bridge or not bridge.is_connected:
            return local_response

        # Dual inference through bridge
        final_response, resonance = bridge.dual_inference(user_input, local_response)

        # Log resonance
        print(f"[Bridge:Adapter] Resonance: {resonance:.2%}")

        return final_response

    def sync_with_remote(self, bridge):
        """
        Synchronize state with remote node

        Args:
            bridge: ResonanceBridge instance
        """
        if not bridge or not bridge.is_connected:
            return

        # Export local state
        local_state = self.export_state()

        # Send to remote
        bridge.sync_state(local_state)

        print(f"[Bridge:Adapter] Synced state with remote node")


# Factory function for clean integration
def create_bridge_adapter(nicole_core_instance):
    """
    Create bridge adapter wrapping Nicole core

    Usage:
        from nicole import nicole_core
        from nicole_bridge_adapter import create_bridge_adapter
        from nicole_bridge import ResonanceBridge

        adapter = create_bridge_adapter(nicole_core)
        bridge = ResonanceBridge(mode='weightless', ...)

        response = adapter.process_with_bridge("hello", bridge)
    """
    return NicoleBridgeAdapter(nicole_core_instance)


if __name__ == "__main__":
    print("=== NICOLE BRIDGE ADAPTER TEST ===")

    # Import core
    from nicole import nicole_core

    # Create adapter
    adapter = create_bridge_adapter(nicole_core)

    # Test export
    print("\n--- Testing state export ---")
    nicole_core.start_conversation("test_bridge_adapter")
    nicole_core.process_message("Hello Nicole")

    state = adapter.export_state()
    print(f"Exported state: {len(state['word_frequencies'])} words, {len(state['bigram_transitions'])} bigrams")

    # Test import
    print("\n--- Testing state import ---")
    mock_state = {
        'word_frequencies': {'test': 10, 'bridge': 5},
        'bigram_transitions': {'test': {'bridge': 3}}
    }
    adapter.import_state(mock_state)

    print("\n✅ Adapter test complete")
