"""
Nicole Bootstrap — Skeleton Loader
Loads all JSON files into memory at startup

STUB: Will be activated after skeleton is exported
"""

import json
from pathlib import Path

SKELETON_DIR = Path("nicole_bootstrap")

_skeleton_cache = None

def load_skeleton():
    """Load all skeleton files (cached)"""
    global _skeleton_cache

    if _skeleton_cache is not None:
        return _skeleton_cache

    print("[Bootstrap] Loading skeleton...")

    skeleton = {}
    for json_file in SKELETON_DIR.glob("*.json"):
        key = json_file.stem  # filename without .json
        try:
            skeleton[key] = json.loads(json_file.read_text())
        except Exception as e:
            print(f"[Bootstrap] Warning: Could not load {json_file}: {e}")

    _skeleton_cache = skeleton
    print(f"[Bootstrap] Loaded {len(skeleton)} skeleton files")

    return skeleton

def get_ngrams():
    """Get n-gram statistics"""
    skel = load_skeleton()
    return skel.get("ngram_stats", {"bigrams": [], "trigrams": []})

def get_shapes():
    """Get phrase shapes"""
    skel = load_skeleton()
    return skel.get("phrase_shapes", {}).get("shapes", [])

def get_clusters():
    """Get semantic clusters"""
    skel = load_skeleton()
    return skel.get("semantic_clusters", {}).get("clusters", {})

def get_style():
    """Get style bias"""
    skel = load_skeleton()
    return skel.get("style_bias", {})

def get_banned():
    """Get banned patterns"""
    skel = load_skeleton()
    return skel.get("banned_patterns", {}).get("patterns", [])

def get_metadata():
    """Get skeleton metadata"""
    skel = load_skeleton()
    return skel.get("metadata", {})
