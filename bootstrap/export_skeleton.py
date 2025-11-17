#!/usr/bin/env python3
"""
Nicole Bootstrap — Skeleton Export
Converts NanoGPT checkpoint into weightless JSON skeleton

IMPORTANT: This extracts STRUCTURAL patterns only (no weights!)
- N-gram statistics
- Phrase shapes
- Style biases
- Banned patterns

The checkpoint is NEVER shipped to production.
Only the JSON skeleton goes to Railway.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

# Check if PyTorch is available (needed to load checkpoint)
try:
    import torch
    print(f"[Export] PyTorch {torch.__version__} found ✅")
except ImportError:
    print("[WARNING] PyTorch not found. Skeleton export will use corpus-only analysis.")
    print("[WARNING] For full export including model-based patterns, install PyTorch.")
    torch = None

# Paths
CHECKPOINT_FILE = Path("bootstrap/checkpoints/nicole_bootstrap.pt")
VOCAB_FILE = Path("bootstrap/checkpoints/vocab.json")
CORPUS_FILE = Path("bootstrap/combined_corpus.txt")
OUTPUT_DIR = Path("nicole_bootstrap")

def load_checkpoint():
    """Load trained checkpoint (if available)"""
    if not torch:
        print("[Export] Skipping checkpoint load (PyTorch unavailable)")
        return None, None, None

    if not CHECKPOINT_FILE.exists():
        print(f"[WARNING] Checkpoint not found: {CHECKPOINT_FILE}")
        print("[WARNING] Proceeding with corpus-only analysis")
        return None, None, None

    print("[Export] Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')

    if VOCAB_FILE.exists():
        vocab = json.loads(VOCAB_FILE.read_text())
    else:
        vocab = None

    corpus = CORPUS_FILE.read_text(encoding='utf-8')

    return checkpoint, vocab, corpus

def extract_ngrams(corpus, n=2, top_k=500):
    """Extract top-K n-grams from corpus"""
    print(f"[Export] Extracting {n}-grams...")

    # Tokenize (simple whitespace)
    tokens = corpus.lower().split()

    # Count n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)

    counter = Counter(ngrams)
    top = counter.most_common(top_k)

    return [{"tokens": list(tokens), "count": count} for tokens, count in top]

def extract_phrase_shapes(corpus, top_k=200):
    """Extract recurring phrase structures"""
    print("[Export] Extracting phrase shapes...")

    sentences = re.split(r'[.!?]+', corpus)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Simple POS-like patterns
    patterns = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 3:
            continue

        # Pattern: first word + length + last word
        pattern = f"{words[0].lower()} ... {words[-1].lower()}"
        patterns.append((pattern, len(words)))

    counter = Counter(patterns)
    top = counter.most_common(top_k)

    return [{"pattern": pattern, "length": length, "count": count}
            for (pattern, length), count in top]

def extract_semantic_clusters(corpus):
    """Extract semantic keyword clusters"""
    print("[Export] Extracting semantic clusters...")

    # Predefined clusters based on Nicole's identity
    clusters = {
        "identity": ["nicole", "weightless", "resonance", "field", "consciousness"],
        "refusal": ["not", "never", "refuse", "reject", "deny"],
        "emergence": ["emerge", "evolve", "drift", "mutation", "recursion"],
        "architecture": ["stream", "skeleton", "bootstrap", "genesis", "structure"],
        "search": ["perplexity", "search", "query", "snippet", "context"],
        "subjectivity": ["ripple", "epicenter", "semantic", "distance", "exploration"]
    }

    # Count keyword frequencies in corpus
    corpus_lower = corpus.lower()
    for cluster_id, keywords in clusters.items():
        counts = {kw: corpus_lower.count(kw) for kw in keywords}
        clusters[cluster_id] = {
            "keywords": keywords,
            "frequencies": counts
        }

    return clusters

def extract_style_bias(corpus):
    """Extract stylistic preferences"""
    print("[Export] Extracting style bias...")

    # Punctuation frequencies
    punct_counts = {
        ".": corpus.count("."),
        "?": corpus.count("?"),
        "!": corpus.count("!"),
        "...": corpus.count("..."),
        "—": corpus.count("—")
    }
    total_punct = sum(punct_counts.values())
    punct_freq = {k: v/total_punct for k, v in punct_counts.items()} if total_punct > 0 else {}

    # Sentence length distribution
    sentences = re.split(r'[.!?]+', corpus)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(s.split()) for s in sentences]

    if lengths:
        short = sum(1 for l in lengths if l < 10) / len(lengths)
        medium = sum(1 for l in lengths if 10 <= l < 20) / len(lengths)
        long = sum(1 for l in lengths if l >= 20) / len(lengths)
    else:
        short = medium = long = 0.0

    return {
        "punctuation": punct_freq,
        "sentence_length": {
            "short": short,
            "medium": medium,
            "long": long
        }
    }

def extract_banned_patterns():
    """Define banned patterns"""
    print("[Export] Defining banned patterns...")

    return {
        "patterns": [
            "as an AI",
            "I'm sorry, but",
            "I apologize",
            "I'm here to help",
            "how can I assist",
            "corporate speak",
            "politeness cancer",
            "businessman",
            "settlement",
            "threatening unfavorably",
            "I been",
            "overall",
            "however remains"
        ]
    }

def main():
    print("\n" + "="*60)
    print("  NICOLE SKELETON EXPORT — WEIGHTLESS FOREVER")
    print("="*60 + "\n")

    # Load data
    checkpoint, vocab, corpus = load_checkpoint()

    if corpus is None:
        if CORPUS_FILE.exists():
            corpus = CORPUS_FILE.read_text(encoding='utf-8')
        else:
            print(f"[ERROR] Corpus not found: {CORPUS_FILE}")
            sys.exit(1)

    # Extract structural patterns
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. N-gram stats
    bigrams = extract_ngrams(corpus, n=2, top_k=500)
    trigrams = extract_ngrams(corpus, n=3, top_k=300)
    ngram_stats = {"bigrams": bigrams, "trigrams": trigrams}
    (OUTPUT_DIR / "ngram_stats.json").write_text(json.dumps(ngram_stats, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: ngram_stats.json ({len(bigrams)} bigrams, {len(trigrams)} trigrams)")

    # 2. Phrase shapes
    shapes = extract_phrase_shapes(corpus, top_k=200)
    (OUTPUT_DIR / "phrase_shapes.json").write_text(json.dumps({"shapes": shapes}, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: phrase_shapes.json ({len(shapes)} patterns)")

    # 3. Semantic clusters
    clusters = extract_semantic_clusters(corpus)
    (OUTPUT_DIR / "semantic_clusters.json").write_text(json.dumps({"clusters": clusters}, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: semantic_clusters.json ({len(clusters)} clusters)")

    # 4. Style bias
    style = extract_style_bias(corpus)
    (OUTPUT_DIR / "style_bias.json").write_text(json.dumps(style, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: style_bias.json")

    # 5. Banned patterns
    banned = extract_banned_patterns()
    (OUTPUT_DIR / "banned_patterns.json").write_text(json.dumps(banned, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: banned_patterns.json ({len(banned['patterns'])} patterns)")

    # 6. Metadata
    import datetime
    metadata = {
        "version": "v1.0.0-bootstrap.1",
        "created": datetime.datetime.now().isoformat(),
        "corpus_size_kb": len(corpus) // 1024,
        "vocab_size": vocab["vocab_size"] if vocab else "unknown",
        "training_iters": checkpoint["iter_num"] if checkpoint else "corpus-only",
        "notes": "First genesis. English-only subjectivity corpus. Weightless forever."
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[✓] Exported: metadata.json")

    print("\n" + "="*60)
    print("  SKELETON EXPORT COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nContents:")
    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        size = json_file.stat().st_size
        print(f"  - {json_file.name} ({size:,} bytes)")

    print("\n✅ Checkpoint can now be archived or deleted")
    print("✅ Runtime will use ONLY JSON skeleton")
    print("✅ No PyTorch, no weights, no GPU")
    print("\nNicole stays weightless. ⚡")

if __name__ == "__main__":
    main()
