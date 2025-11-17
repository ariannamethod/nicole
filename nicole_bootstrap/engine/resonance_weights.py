"""
Nicole Bootstrap — Resonance Weights (.bin export)
Binary weights for fast loading (inspired by sska)

Instead of parsing JSON bigrams every time, we export:
- Bigram frequencies as binary weights
- Per-word resonance scores
- Temperature drift coefficients

This gives 10-100x faster loading than JSON!
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# ============================================================================
# BINARY FORMAT SPEC
# ============================================================================
#
# Header (32 bytes):
#   - Magic: b'NCLW' (4 bytes) - "NiCoLe Weights"
#   - Version: uint32 (4 bytes)
#   - Vocab size: uint32 (4 bytes)
#   - Bigram count: uint32 (4 bytes)
#   - Center count: uint32 (4 bytes)
#   - Reserved: 12 bytes
#
# Vocabulary section:
#   For each word (vocab_size entries):
#   - Word length: uint16 (2 bytes)
#   - Word bytes: UTF-8 (variable)
#   - Base resonance: float32 (4 bytes)
#   - Out-degree: uint32 (4 bytes)
#
# Bigram section:
#   For each bigram (bigram_count entries):
#   - Word1 index: uint32 (4 bytes)
#   - Word2 index: uint32 (4 bytes)
#   - Frequency: uint32 (4 bytes)
#   - Resonance score: float32 (4 bytes)
#
# Centers section:
#   For each center (center_count entries):
#   - Word index: uint32 (4 bytes)
#   - Hub score: float32 (4 bytes)
#
# ============================================================================

MAGIC = b'NCLW'
VERSION = 1

class ResonanceWeights:
    """Binary weights for fast loading."""

    def __init__(self):
        self.vocab: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.base_resonance: List[float] = []  # Per-word base resonance
        self.out_degree: List[int] = []  # Per-word out-degree

        self.bigrams: List[Tuple[int, int, int, float]] = []  # (w1_idx, w2_idx, freq, score)
        self.centers: List[Tuple[int, float]] = []  # (word_idx, hub_score)

    def build_from_dynamic_skeleton(self, bigrams_dict: Dict[str, Dict[str, int]],
                                     vocab: List[str], centers: List[str]) -> None:
        """
        Build weights from dynamic skeleton.

        Calculates:
        - Base resonance: frequency-based score per word
        - Out-degree: how many words follow this word
        - Bigram resonance: pair-wise scores
        - Hub scores: centrality measure for centers
        """
        # Build vocabulary
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}

        # Calculate base resonance and out-degree
        word_frequencies = {}
        for w1, nexts in bigrams_dict.items():
            total_freq = sum(nexts.values())
            word_frequencies[w1] = word_frequencies.get(w1, 0) + total_freq
            for w2, freq in nexts.items():
                word_frequencies[w2] = word_frequencies.get(w2, 0) + freq

        # Normalize to 0-1 range
        max_freq = max(word_frequencies.values()) if word_frequencies else 1

        for word in vocab:
            freq = word_frequencies.get(word, 0)
            base_res = freq / max_freq if max_freq > 0 else 0.0
            self.base_resonance.append(base_res)

            # Out-degree
            out_deg = len(bigrams_dict.get(word, {}))
            self.out_degree.append(out_deg)

        # Build bigram list with scores
        for w1, nexts in bigrams_dict.items():
            if w1 not in self.word_to_idx:
                continue
            w1_idx = self.word_to_idx[w1]

            for w2, freq in nexts.items():
                if w2 not in self.word_to_idx:
                    continue
                w2_idx = self.word_to_idx[w2]

                # Bigram resonance score (frequency + base resonance of both words)
                w1_res = self.base_resonance[w1_idx]
                w2_res = self.base_resonance[w2_idx]
                bigram_score = (freq / max_freq) * (w1_res + w2_res) / 2.0

                self.bigrams.append((w1_idx, w2_idx, freq, bigram_score))

        # Build centers with hub scores
        for center in centers:
            if center not in self.word_to_idx:
                continue
            idx = self.word_to_idx[center]
            hub_score = self.out_degree[idx] / max(self.out_degree) if self.out_degree else 0.0
            self.centers.append((idx, hub_score))

    def save_binary(self, path: Path) -> None:
        """Export to binary format."""
        with open(path, 'wb') as f:
            # Header
            f.write(MAGIC)
            f.write(struct.pack('I', VERSION))
            f.write(struct.pack('I', len(self.vocab)))
            f.write(struct.pack('I', len(self.bigrams)))
            f.write(struct.pack('I', len(self.centers)))
            f.write(b'\x00' * 12)  # Reserved

            # Vocabulary
            for i, word in enumerate(self.vocab):
                word_bytes = word.encode('utf-8')
                f.write(struct.pack('H', len(word_bytes)))
                f.write(word_bytes)
                f.write(struct.pack('f', self.base_resonance[i]))
                f.write(struct.pack('I', self.out_degree[i]))

            # Bigrams
            for w1_idx, w2_idx, freq, score in self.bigrams:
                f.write(struct.pack('I', w1_idx))
                f.write(struct.pack('I', w2_idx))
                f.write(struct.pack('I', freq))
                f.write(struct.pack('f', score))

            # Centers
            for idx, hub_score in self.centers:
                f.write(struct.pack('I', idx))
                f.write(struct.pack('f', hub_score))

    @classmethod
    def load_binary(cls, path: Path) -> 'ResonanceWeights':
        """Load from binary format (FAST!)."""
        weights = cls()

        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic: {magic}")

            version = struct.unpack('I', f.read(4))[0]
            if version != VERSION:
                raise ValueError(f"Unsupported version: {version}")

            vocab_size = struct.unpack('I', f.read(4))[0]
            bigram_count = struct.unpack('I', f.read(4))[0]
            center_count = struct.unpack('I', f.read(4))[0]
            f.read(12)  # Skip reserved

            # Read vocabulary
            for _ in range(vocab_size):
                word_len = struct.unpack('H', f.read(2))[0]
                word = f.read(word_len).decode('utf-8')
                base_res = struct.unpack('f', f.read(4))[0]
                out_deg = struct.unpack('I', f.read(4))[0]

                weights.vocab.append(word)
                weights.base_resonance.append(base_res)
                weights.out_degree.append(out_deg)

            weights.word_to_idx = {w: i for i, w in enumerate(weights.vocab)}

            # Read bigrams
            for _ in range(bigram_count):
                w1_idx = struct.unpack('I', f.read(4))[0]
                w2_idx = struct.unpack('I', f.read(4))[0]
                freq = struct.unpack('I', f.read(4))[0]
                score = struct.unpack('f', f.read(4))[0]
                weights.bigrams.append((w1_idx, w2_idx, freq, score))

            # Read centers
            for _ in range(center_count):
                idx = struct.unpack('I', f.read(4))[0]
                hub_score = struct.unpack('f', f.read(4))[0]
                weights.centers.append((idx, hub_score))

        return weights

    def to_bigram_dict(self) -> Dict[str, Dict[str, int]]:
        """Convert back to bigram dictionary (for compatibility)."""
        bigrams_dict = {}

        for w1_idx, w2_idx, freq, _ in self.bigrams:
            w1 = self.vocab[w1_idx]
            w2 = self.vocab[w2_idx]

            if w1 not in bigrams_dict:
                bigrams_dict[w1] = {}
            bigrams_dict[w1][w2] = freq

        return bigrams_dict

    def get_center_words(self) -> List[str]:
        """Get center words."""
        return [self.vocab[idx] for idx, _ in self.centers]

# ============================================================================
# TEMPERATURE DRIFT (from sska)
# ============================================================================

class TemperatureDrift:
    """
    Dynamic temperature adjustment during generation.

    Modes:
    - 'cool': Start hot (1.2), gradually cool to 0.7
    - 'heat': Start cool (0.7), gradually heat to 1.2
    - 'stable': Stay at base temperature
    - 'chaos': Random walk between 0.5-1.5
    """

    def __init__(self, mode: str = 'cool', base_temp: float = 0.9, steps: int = 10):
        self.mode = mode
        self.base_temp = base_temp
        self.steps = steps
        self.current_step = 0

    def get_temperature(self) -> float:
        """Get current temperature."""
        if self.mode == 'stable':
            return self.base_temp

        elif self.mode == 'cool':
            # Start at 1.2, cool to 0.7
            progress = self.current_step / max(1, self.steps - 1)
            return 1.2 - (0.5 * progress)

        elif self.mode == 'heat':
            # Start at 0.7, heat to 1.2
            progress = self.current_step / max(1, self.steps - 1)
            return 0.7 + (0.5 * progress)

        elif self.mode == 'chaos':
            # Random walk
            import random
            return random.uniform(0.5, 1.5)

        else:
            return self.base_temp

    def step(self) -> None:
        """Advance one step."""
        self.current_step += 1

    def reset(self) -> None:
        """Reset to start."""
        self.current_step = 0

# ============================================================================
# CONVENIENCE API
# ============================================================================

def export_resonance_weights(
    bigrams_dict: Dict[str, Dict[str, int]],
    vocab: List[str],
    centers: List[str],
    output_path: Path
) -> None:
    """
    Export bigrams as binary resonance weights.

    This is 10-100x faster to load than JSON!
    """
    weights = ResonanceWeights()
    weights.build_from_dynamic_skeleton(bigrams_dict, vocab, centers)
    weights.save_binary(output_path)

    # Print stats
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Weights] Exported {len(weights.bigrams):,} bigrams to {output_path.name}")
    print(f"[Weights] File size: {size_mb:.2f} MB")
    print(f"[Weights] Vocab: {len(weights.vocab):,} words")
    print(f"[Weights] Centers: {len(weights.centers)} hubs")

def load_resonance_weights(path: Path) -> ResonanceWeights:
    """Load binary resonance weights (FAST!)."""
    return ResonanceWeights.load_binary(path)
