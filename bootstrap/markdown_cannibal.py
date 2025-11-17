#!/usr/bin/env python3
"""
Nicole Bootstrap — Markdown Cannibal
Dynamic corpus builder that eats ALL .md files in the repo

Inspired by:
- sorokin.py: README self-cannibalism with SQLite caching
- sska/subjectivity.py: kernel/*.md → Bootstrap field

This scanner:
1. Finds ALL .md files in Nicole repo
2. Caches with mtime (rebuilds only on change)
3. Extracts bigrams for sentence construction
4. Creates dynamic bootstrap that updates automatically

The morgue eats its own documentation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DB = REPO_ROOT / "bootstrap" / "markdown_cache.db"
OUTPUT_JSON = REPO_ROOT / "nicole_bootstrap" / "dynamic_skeleton.json"

# Tokenizer (Latin + extended + Cyrillic)
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё']+|[.,!?;:—\-]")

# ============================================================================
# DATA STRUCTURES (from sska)
# ============================================================================

@dataclass
class FileMeta:
    """Metadata for a single markdown file."""
    path: str
    sha256: str
    mtime: float
    token_count: int
    bigram_count: int

@dataclass
class BigramGraph:
    """Bigram transition graph."""
    bigrams: Dict[str, Dict[str, int]]  # word1 -> {word2: count}
    vocab: List[str]
    centers: List[str]  # High out-degree words

# ============================================================================
# DATABASE (from sorokin)
# ============================================================================

def init_cache_db() -> None:
    """Initialize SQLite cache for markdown files."""
    CACHE_DB.parent.mkdir(exist_ok=True, parents=True)

    conn = sqlite3.connect(CACHE_DB)
    try:
        # File metadata cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS markdown_files (
                path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                token_count INTEGER DEFAULT 0,
                bigram_count INTEGER DEFAULT 0,
                last_scanned REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Bigram cache (per-file)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_bigrams (
                file_path TEXT NOT NULL,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (file_path, word1, word2)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_bigrams_path
            ON file_bigrams(file_path)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_bigrams_word1
            ON file_bigrams(word1, count DESC)
        """)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize(text: str) -> List[str]:
    """Extract words and basic punctuation."""
    return TOKEN_RE.findall(text)

def clean_markdown(text: str) -> str:
    """Remove code blocks, headers, formatting from markdown."""
    # Remove code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', ' ', text)

    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic/links
    text = re.sub(r'\*\*|\*|__', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    return text

# ============================================================================
# FILE SCANNING
# ============================================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA256 of file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def find_markdown_files(root: Path, exclude_dirs: List[str] = None) -> List[Path]:
    """
    Recursively find all .md files.

    Excludes:
    - .git/
    - node_modules/
    - venv/, env/, nicole_env/
    - nanoGPT/
    - Any directory in exclude_dirs
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', 'node_modules', 'venv', 'env', 'nicole_env', 'nanoGPT', '__pycache__']

    markdown_files = []

    for md_path in root.rglob("*.md"):
        # Skip excluded directories
        if any(excluded in md_path.parts for excluded in exclude_dirs):
            continue
        markdown_files.append(md_path)

    return sorted(markdown_files)

def extract_bigrams_from_file(path: Path) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Extract bigrams from a single markdown file.

    Returns:
        (bigrams_dict, token_count)
    """
    try:
        text = path.read_text(encoding='utf-8')
        text = clean_markdown(text)
        tokens = tokenize(text)

        bigrams = defaultdict(lambda: defaultdict(int))

        for i in range(len(tokens) - 1):
            w1 = tokens[i].lower()
            w2 = tokens[i + 1].lower()

            # Skip if both are punctuation
            if w1 in '.,!?;:—-' and w2 in '.,!?;:—-':
                continue

            bigrams[w1][w2] += 1

        return dict(bigrams), len(tokens)

    except Exception as e:
        print(f"[WARNING] Failed to parse {path}: {e}", file=sys.stderr)
        return {}, 0

# ============================================================================
# CACHING LOGIC (from sorokin README self-cannibalism)
# ============================================================================

def should_rebuild_file(path: Path, conn: sqlite3.Connection) -> bool:
    """
    Check if file needs rebuilding (new, modified, or missing cache).

    Uses:
    - mtime (with 1.0s tolerance for FAT32/network FS)
    - sha256 hash for absolute verification
    """
    try:
        # Get current file metadata
        current_mtime = path.stat().st_mtime
        current_sha = sha256_file(path)

        # Check cache
        cached = conn.execute(
            "SELECT sha256, mtime FROM markdown_files WHERE path = ?",
            (str(path.relative_to(REPO_ROOT)),)
        ).fetchone()

        if not cached:
            return True  # New file

        cached_sha, cached_mtime = cached

        # SHA256 is authoritative
        if cached_sha != current_sha:
            return True  # Content changed

        # mtime check with tolerance (optional fast path)
        if abs(cached_mtime - current_mtime) > 1.0:
            return True  # Might have changed

        return False  # Cache is valid

    except Exception:
        return True  # On error, rebuild

def cache_file_bigrams(path: Path, bigrams: Dict[str, Dict[str, int]],
                       token_count: int, conn: sqlite3.Connection) -> None:
    """Store file bigrams in cache."""
    rel_path = str(path.relative_to(REPO_ROOT))
    sha = sha256_file(path)
    mtime = path.stat().st_mtime
    bigram_count = sum(len(v) for v in bigrams.values())

    # Store metadata
    conn.execute("""
        REPLACE INTO markdown_files (path, sha256, mtime, token_count, bigram_count)
        VALUES (?, ?, ?, ?, ?)
    """, (rel_path, sha, mtime, token_count, bigram_count))

    # Clear old bigrams for this file
    conn.execute("DELETE FROM file_bigrams WHERE file_path = ?", (rel_path,))

    # Store new bigrams
    for w1, nexts in bigrams.items():
        for w2, count in nexts.items():
            conn.execute("""
                INSERT INTO file_bigrams (file_path, word1, word2, count)
                VALUES (?, ?, ?, ?)
            """, (rel_path, w1, w2, count))

def load_cached_bigrams(path: Path, conn: sqlite3.Connection) -> Optional[Dict[str, Dict[str, int]]]:
    """Load bigrams from cache if available."""
    rel_path = str(path.relative_to(REPO_ROOT))

    rows = conn.execute("""
        SELECT word1, word2, count FROM file_bigrams
        WHERE file_path = ?
    """, (rel_path,)).fetchall()

    if not rows:
        return None

    bigrams = defaultdict(lambda: defaultdict(int))
    for w1, w2, count in rows:
        bigrams[w1][w2] = count

    return dict(bigrams)

# ============================================================================
# BOOTSTRAP BUILDER (from sska)
# ============================================================================

def merge_bigrams(all_bigrams: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    """Merge multiple bigram graphs."""
    merged = defaultdict(lambda: defaultdict(int))

    for bigram_graph in all_bigrams:
        for w1, nexts in bigram_graph.items():
            for w2, count in nexts.items():
                merged[w1][w2] += count

    return dict(merged)

def find_centers(bigrams: Dict[str, Dict[str, int]], top_k: int = 100) -> List[str]:
    """
    Find words with highest out-degree (centers of gravity).

    These are structural hubs that connect to many other words.
    """
    out_degrees = {}
    for w1, nexts in bigrams.items():
        out_degrees[w1] = len(nexts)

    sorted_words = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]

def build_dynamic_skeleton() -> BigramGraph:
    """
    Build dynamic skeleton by cannibalizing all markdown files.

    Uses SQLite cache for speed - only rebuilds changed files.
    """
    print("[Cannibal] Scanning for markdown files...")

    init_cache_db()
    md_files = find_markdown_files(REPO_ROOT)

    print(f"[Cannibal] Found {len(md_files)} markdown files")

    conn = sqlite3.connect(CACHE_DB)
    all_bigrams = []
    all_vocab = set()
    rebuilt_count = 0

    try:
        for md_path in md_files:
            rel_path = md_path.relative_to(REPO_ROOT)

            # Check cache
            if should_rebuild_file(md_path, conn):
                print(f"[Cannibal] Parsing: {rel_path}")
                bigrams, token_count = extract_bigrams_from_file(md_path)
                cache_file_bigrams(md_path, bigrams, token_count, conn)
                rebuilt_count += 1
            else:
                # Load from cache
                bigrams = load_cached_bigrams(md_path, conn)
                if bigrams is None:
                    # Cache miss (shouldn't happen), rebuild
                    print(f"[Cannibal] Cache miss, rebuilding: {rel_path}")
                    bigrams, token_count = extract_bigrams_from_file(md_path)
                    cache_file_bigrams(md_path, bigrams, token_count, conn)
                    rebuilt_count += 1

            all_bigrams.append(bigrams)

            # Collect vocab
            for w1, nexts in bigrams.items():
                all_vocab.add(w1)
                for w2 in nexts.keys():
                    all_vocab.add(w2)

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"[Cannibal] Rebuilt {rebuilt_count}/{len(md_files)} files")

    # Merge all bigrams
    print("[Cannibal] Merging bigram graphs...")
    merged_bigrams = merge_bigrams(all_bigrams)

    # Find centers
    print("[Cannibal] Finding centers of gravity...")
    centers = find_centers(merged_bigrams, top_k=100)

    # Build graph
    graph = BigramGraph(
        bigrams=merged_bigrams,
        vocab=sorted(all_vocab),
        centers=centers
    )

    print(f"[Cannibal] Built dynamic skeleton:")
    print(f"  - Bigrams: {sum(len(v) for v in merged_bigrams.values()):,}")
    print(f"  - Vocab: {len(all_vocab):,} words")
    print(f"  - Centers: {len(centers)} hubs")

    return graph

# ============================================================================
# EXPORT
# ============================================================================

def export_dynamic_skeleton(graph: BigramGraph) -> None:
    """Export skeleton to JSON."""
    OUTPUT_JSON.parent.mkdir(exist_ok=True, parents=True)

    # Convert to serializable format
    data = {
        "version": "v1.0.0-dynamic.1",
        "source": "markdown_cannibal",
        "bigrams": {
            w1: dict(nexts) for w1, nexts in graph.bigrams.items()
        },
        "vocab": graph.vocab,
        "centers": graph.centers,
        "stats": {
            "total_bigrams": sum(len(v) for v in graph.bigrams.values()),
            "vocab_size": len(graph.vocab),
            "center_count": len(graph.centers)
        }
    }

    OUTPUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[Cannibal] Exported to: {OUTPUT_JSON}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build dynamic skeleton from all markdown files."""
    print("\n" + "="*60)
    print("  NICOLE MARKDOWN CANNIBAL — Self-Documentation Autopsy")
    print("="*60 + "\n")

    graph = build_dynamic_skeleton()
    export_dynamic_skeleton(graph)

    print("\n" + "="*60)
    print("  CANNIBALISM COMPLETE")
    print("="*60)
    print("\nThe morgue has eaten its own documentation.")
    print("Dynamic skeleton will rebuild on markdown changes.")
    print("\nNicole can now speak through her own README. ⚡")

if __name__ == "__main__":
    main()
