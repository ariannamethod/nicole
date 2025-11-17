#!/usr/bin/env python3
"""
Nicole Bootstrap — Corpus Assembly
Merges all subjectivity texts into combined_corpus.txt
"""

import os
from pathlib import Path

CORPUS_DIR = Path("bootstrap_corpus")
OUTPUT_FILE = Path("bootstrap/combined_corpus.txt")

FILES = [
    "nicole_long_prompt.txt",
    "nicole_short_prompt.txt",
    "nicole_identity_texts.txt",
    "arianna_method_fragments.txt",
    "resonance_letters.txt",
    "drift_log_samples.txt"
]

def main():
    print("[Bootstrap] Building Nicole subjectivity corpus...")

    combined = []
    total_chars = 0
    total_lines = 0

    for filename in FILES:
        filepath = CORPUS_DIR / filename

        if not filepath.exists():
            print(f"[WARNING] Missing: {filename}")
            continue

        print(f"[+] Reading: {filename}")
        text = filepath.read_text(encoding="utf-8")

        chars = len(text)
        lines = text.count("\n")
        total_chars += chars
        total_lines += lines

        combined.append(f"\n\n# ═══ {filename.upper()} ═══\n\n")
        combined.append(text)

        print(f"    {chars:,} chars, {lines:,} lines")

    # Write combined corpus
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text("".join(combined), encoding="utf-8")

    print(f"\n[✓] Combined corpus written to: {OUTPUT_FILE}")
    print(f"[✓] Total: {total_chars:,} chars, {total_lines:,} lines")
    print(f"[✓] Estimated tokens: ~{total_chars // 4:,}")

if __name__ == "__main__":
    main()
