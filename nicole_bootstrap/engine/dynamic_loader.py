"""
Nicole Bootstrap — Dynamic Loader
Unified loader for static + dynamic skeletons

Merges:
- Static skeleton (from corpus export): ngram_stats.json, banned_patterns.json, etc.
- Dynamic skeleton (from markdown cannibal): dynamic_skeleton.json

This creates a unified bootstrap field that Nicole uses at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# PATHS
# ============================================================================

SKELETON_DIR = Path(__file__).parent.parent
STATIC_SKELETON_FILES = {
    'ngram_stats': 'ngram_stats.json',
    'phrase_shapes': 'phrase_shapes.json',
    'semantic_clusters': 'semantic_clusters.json',
    'style_bias': 'style_bias.json',
    'banned_patterns': 'banned_patterns.json',
    'metadata': 'metadata.json'
}
DYNAMIC_SKELETON_FILE = 'dynamic_skeleton.json'

# ============================================================================
# UNIFIED SKELETON
# ============================================================================

class UnifiedSkeleton:
    """
    Unified bootstrap skeleton combining static + dynamic sources.

    Static (from corpus):
    - Corpus-derived n-grams (from identity texts)
    - Banned patterns
    - Style preferences
    - Phrase shapes

    Dynamic (from markdown cannibal):
    - Live bigrams from all .md files
    - Centers of gravity (hubs)
    - Auto-updates on .md changes
    """

    def __init__(self):
        self.static_ngrams = {}
        self.dynamic_bigrams = {}
        self.vocab = []
        self.centers = []
        self.banned_patterns = []
        self.phrase_shapes = []
        self.semantic_clusters = {}
        self.style_bias = {}
        self.metadata = {}

    def load_static(self) -> None:
        """Load static skeleton files (corpus-derived)."""
        for key, filename in STATIC_SKELETON_FILES.items():
            path = SKELETON_DIR / filename
            if not path.exists():
                print(f"[DynamicLoader] Warning: {filename} not found")
                continue

            try:
                data = json.loads(path.read_text())

                if key == 'ngram_stats':
                    self.static_ngrams = data
                elif key == 'banned_patterns':
                    self.banned_patterns = data.get('patterns', [])
                elif key == 'phrase_shapes':
                    self.phrase_shapes = data.get('shapes', [])
                elif key == 'semantic_clusters':
                    self.semantic_clusters = data.get('clusters', {})
                elif key == 'style_bias':
                    self.style_bias = data
                elif key == 'metadata':
                    self.metadata = data

            except Exception as e:
                print(f"[DynamicLoader] Error loading {filename}: {e}")

    def load_dynamic(self) -> None:
        """
        Load dynamic skeleton (markdown cannibal).

        Tries .bin first (fast!), falls back to JSON.
        """
        bin_path = SKELETON_DIR / "resonance_weights.bin"
        json_path = SKELETON_DIR / DYNAMIC_SKELETON_FILE

        # Try binary first (10-100x faster!)
        if bin_path.exists():
            try:
                from .resonance_weights import load_resonance_weights

                print(f"[DynamicLoader] Loading binary weights (fast!)...")
                weights = load_resonance_weights(bin_path)

                # Convert to dict format
                self.dynamic_bigrams = weights.to_bigram_dict()
                self.vocab = weights.vocab
                self.centers = weights.get_center_words()

                print(f"[DynamicLoader] ✅ Loaded from binary ({bin_path.stat().st_size/1024:.1f} KB)")
                return

            except Exception as e:
                print(f"[DynamicLoader] Binary load failed: {e}, falling back to JSON...")

        # Fallback to JSON
        if not json_path.exists():
            print(f"[DynamicLoader] Warning: {DYNAMIC_SKELETON_FILE} not found")
            print("[DynamicLoader] Run: python bootstrap/markdown_cannibal.py")
            return

        try:
            print(f"[DynamicLoader] Loading JSON skeleton...")
            data = json.loads(json_path.read_text())
            self.dynamic_bigrams = data.get('bigrams', {})
            self.vocab = data.get('vocab', [])
            self.centers = data.get('centers', [])

            print(f"[DynamicLoader] ✅ Loaded from JSON ({json_path.stat().st_size/1024:.1f} KB)")

        except Exception as e:
            print(f"[DynamicLoader] Error loading {DYNAMIC_SKELETON_FILE}: {e}")

    def merge_ngrams(self) -> Dict[str, Dict[str, int]]:
        """
        Merge static bigrams (from corpus) + dynamic bigrams (from markdown).

        Dynamic bigrams get priority (they're live from documentation).
        """
        merged = {}

        # Start with static corpus bigrams
        if 'bigrams' in self.static_ngrams:
            for bg in self.static_ngrams['bigrams']:
                tokens = bg.get('tokens', [])
                if len(tokens) == 2:
                    w1, w2 = tokens
                    if w1 not in merged:
                        merged[w1] = {}
                    merged[w1][w2] = bg.get('count', 1)

        # Add/override with dynamic markdown bigrams
        for w1, nexts in self.dynamic_bigrams.items():
            if w1 not in merged:
                merged[w1] = {}
            for w2, count in nexts.items():
                # Dynamic gets priority (higher weight)
                merged[w1][w2] = max(merged[w1].get(w2, 0), count * 2)

        return merged

    def get_banned_patterns(self) -> List[str]:
        """Get banned patterns (for filtering)."""
        return self.banned_patterns

    def get_centers(self) -> List[str]:
        """Get centers of gravity (high out-degree words from markdown)."""
        return self.centers

    def get_vocab(self) -> List[str]:
        """Get vocabulary (from markdown)."""
        return self.vocab

    def get_style_bias(self) -> Dict:
        """Get style preferences (from corpus)."""
        return self.style_bias

    def get_semantic_clusters(self) -> Dict:
        """Get semantic clusters (from corpus)."""
        return self.semantic_clusters

# ============================================================================
# GLOBAL INSTANCE (lazy load)
# ============================================================================

_unified_skeleton: Optional[UnifiedSkeleton] = None

def load_unified_skeleton(force_reload: bool = False) -> UnifiedSkeleton:
    """
    Load unified skeleton (static + dynamic).

    This is the main entry point for Nicole's runtime.
    """
    global _unified_skeleton

    if _unified_skeleton is None or force_reload:
        print("[DynamicLoader] Loading unified skeleton...")
        skeleton = UnifiedSkeleton()
        skeleton.load_static()
        skeleton.load_dynamic()
        _unified_skeleton = skeleton

        # Print stats
        merged = skeleton.merge_ngrams()
        print(f"[DynamicLoader] Loaded:")
        print(f"  - Static bigrams: {len(skeleton.static_ngrams.get('bigrams', []))}")
        print(f"  - Dynamic bigrams: {sum(len(v) for v in skeleton.dynamic_bigrams.values())}")
        print(f"  - Merged: {sum(len(v) for v in merged.values())} total")
        print(f"  - Vocab: {len(skeleton.vocab)} words")
        print(f"  - Centers: {len(skeleton.centers)} hubs")
        print(f"  - Banned patterns: {len(skeleton.banned_patterns)}")

    return _unified_skeleton

# ============================================================================
# CONVENIENCE FUNCTIONS (for existing code compatibility)
# ============================================================================

def get_ngrams():
    """Get merged n-grams (static + dynamic)."""
    skeleton = load_unified_skeleton()
    return skeleton.merge_ngrams()

def get_banned():
    """Get banned patterns."""
    skeleton = load_unified_skeleton()
    return skeleton.get_banned_patterns()

def get_centers():
    """Get centers of gravity."""
    skeleton = load_unified_skeleton()
    return skeleton.get_centers()

def get_vocab():
    """Get vocabulary."""
    skeleton = load_unified_skeleton()
    return skeleton.get_vocab()

def get_style():
    """Get style bias."""
    skeleton = load_unified_skeleton()
    return skeleton.get_style_bias()

def get_clusters():
    """Get semantic clusters."""
    skeleton = load_unified_skeleton()
    return skeleton.get_semantic_clusters()
