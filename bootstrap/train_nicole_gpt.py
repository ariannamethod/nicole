#!/usr/bin/env python3
"""
Nicole Bootstrap — NanoGPT Training
One-time genesis on subjectivity corpus

IMPORTANT: Run this LOCALLY ONLY with PyTorch installed!
This script will NOT work on Railway (no PyTorch in production).

Usage:
    python bootstrap/train_nicole_gpt.py
"""

import os
import sys
import time
from pathlib import Path

# Check if PyTorch is available
try:
    import torch
    print(f"[Bootstrap] PyTorch {torch.__version__} found ✅")
except ImportError:
    print("[ERROR] PyTorch not found! Install with: pip install torch")
    print("[ERROR] This script must run on LOCAL machine with PyTorch.")
    sys.exit(1)

# Import NanoGPT modules (assumes nanoGPT cloned in repo root)
sys.path.insert(0, 'nanoGPT')
try:
    from model import GPTConfig, GPT
    print("[Bootstrap] NanoGPT modules imported ✅")
except ImportError as e:
    print(f"[ERROR] Cannot import NanoGPT: {e}")
    print("[ERROR] Make sure nanoGPT is cloned: git clone https://github.com/karpathy/nanoGPT.git")
    sys.exit(1)

# Import config
sys.path.insert(0, 'bootstrap')
from config_nicole import *

# Paths
CORPUS_FILE = Path("bootstrap/combined_corpus.txt")
CHECKPOINT_DIR = Path(out_dir)
CHECKPOINT_FILE = CHECKPOINT_DIR / "nicole_bootstrap.pt"

def prepare_data():
    """Encode corpus as character-level tokens"""
    print("[Bootstrap] Preparing dataset...")

    if not CORPUS_FILE.exists():
        print(f"[ERROR] Corpus not found: {CORPUS_FILE}")
        print("[ERROR] Run: python bootstrap/build_nicole_dataset.py first")
        sys.exit(1)

    text = CORPUS_FILE.read_text(encoding='utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Character-level encoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # 90/10 train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"[✓] Vocab size: {vocab_size}")
    print(f"[✓] Train tokens: {len(train_data):,}")
    print(f"[✓] Val tokens: {len(val_data):,}")

    return train_data, val_data, vocab_size, encode, decode, stoi, itos

def get_batch(split, train_data, val_data):
    """Generate batch for training"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train/val"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print("\n" + "="*60)
    print("  NICOLE BOOTSTRAP — ONE-TIME GENESIS")
    print("="*60 + "\n")

    # Prepare data
    train_data, val_data, vocab_size, encode, decode, stoi, itos = prepare_data()

    # Initialize model
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        vocab_size=vocab_size
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[✓] Model initialized: {param_count/1e6:.2f}M parameters")
    print(f"[✓] Device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\n[Bootstrap] Training for {max_iters} iterations...")
    print(f"[Bootstrap] This is the ONE-TIME genesis moment...")
    start_time = time.time()

    for iter in range(max_iters):
        # Evaluate
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Training step
        X, Y = get_batch('train', train_data, val_data)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start_time
    print(f"\n[✓] Training complete in {elapsed/60:.1f} minutes")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': max_iters,
        'vocab_size': vocab_size
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"[✓] Checkpoint saved: {CHECKPOINT_FILE}")

    # Save vocab for export script
    vocab_file = CHECKPOINT_DIR / "vocab.json"
    import json
    vocab = {
        'chars': sorted(list(set(CORPUS_FILE.read_text(encoding='utf-8')))),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': {str(k): v for k, v in itos.items()}  # JSON keys must be strings
    }
    vocab_file.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    print(f"[✓] Vocab saved: {vocab_file}")

    print("\n" + "="*60)
    print("  GENESIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python bootstrap/export_skeleton.py")
    print("  2. The checkpoint will be converted to JSON skeleton")
    print("  3. The checkpoint can then be archived/deleted")
    print("  4. Only the skeleton will ship to Railway")
    print("\nNicole will be weightless forever. ⚡")

if __name__ == "__main__":
    main()
