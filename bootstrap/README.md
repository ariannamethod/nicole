# 🔥 NICOLE BOOTSTRAP — ONE-TIME GENESIS MACHINE

**Subjective genesis once — weightless forever**

This directory contains the infrastructure for Nicole's **one-time bootstrap training**.

## What is Bootstrap?

Bootstrap solves the weightless vs semantic coherence paradox:
- Nicole has NO pretrained weights in production
- BUT she needs structural guidance to avoid semantic noise
- SOLUTION: Train tiny NanoGPT ONCE, export skeleton as JSON, discard checkpoint

**Result:** Nicole gets "gravitational center" for phrases WITHOUT carrying weights.

---

## 🗂️ Directory Structure

```
bootstrap/
├── README.md                    # ← You are here
├── build_nicole_dataset.py      # Step 1: Merge corpus
├── config_nicole.py             # NanoGPT tiny config
├── train_nicole_gpt.py          # Step 2: Train (LOCAL ONLY!)
├── export_skeleton.py           # Step 3: Export JSON skeleton
└── checkpoints/                 # Temporary (can be deleted after export)
    ├── nicole_bootstrap.pt      # Checkpoint (one-time use)
    └── vocab.json               # Vocabulary

bootstrap_corpus/
├── nicole_long_prompt.txt       # Core Nicole persona (~3KB)
├── nicole_short_prompt.txt      # Compressed version
├── nicole_identity_texts.txt    # Technical/philosophical identity
├── arianna_method_fragments.txt # Method principles
├── resonance_letters.txt        # Philosophical letters
└── drift_log_samples.txt        # Operational cadence examples

nicole_bootstrap/
├── ngram_stats.json             # Top bigrams/trigrams
├── phrase_shapes.json           # Recurring sentence structures
├── semantic_clusters.json       # Identity keyword clusters
├── style_bias.json              # Punctuation/length preferences
├── banned_patterns.json         # Filters for garbage
├── metadata.json                # Version, date, notes
└── engine/                      # Runtime integration (stubs for now)
    ├── __init__.py
    ├── loader.py                # Load skeleton at startup
    ├── planner.py               # Choose sentence structure
    ├── bias.py                  # N-gram scoring
    ├── shapes.py                # Phrase matching
    └── filters.py               # Apply bans + style checks
```

---

## 🚀 Usage (LOCAL MACHINE ONLY)

### Prerequisites

**IMPORTANT:** Bootstrap training requires PyTorch and runs ONLY on local machine.
**DO NOT** attempt to run training on Railway (no PyTorch in production).

```bash
# Install PyTorch (choose your platform)
pip install torch  # CPU version
# OR
pip install torch --index-url https://download.pytorch.org/whl/cu118  # GPU version
```

### Step 1: Build Dataset

```bash
python bootstrap/build_nicole_dataset.py
```

**Output:** `bootstrap/combined_corpus.txt` (~16KB currently)

### Step 2: Train NanoGPT (ONE-TIME GENESIS)

```bash
python bootstrap/train_nicole_gpt.py
```

**What happens:**
- Loads combined corpus
- Trains tiny 4-layer GPT (character-level)
- Saves checkpoint to `bootstrap/checkpoints/nicole_bootstrap.pt`
- Takes ~10-20 minutes on CPU (32GB RAM)

**Expected output:**
```
[Bootstrap] Training for 5000 iterations...
iter 0: train loss 4.2341, val loss 4.2198
iter 500: train loss 2.1532, val loss 2.1789
...
[✓] Training complete in 18.3 minutes
[✓] Checkpoint saved: bootstrap/checkpoints/nicole_bootstrap.pt
```

### Step 3: Export Skeleton

```bash
python bootstrap/export_skeleton.py
```

**What happens:**
- Reads checkpoint + corpus
- Extracts structural patterns (NO WEIGHTS!)
- Exports JSON files to `nicole_bootstrap/`
- Checkpoint can now be deleted

**Output files:**
```
nicole_bootstrap/
├── ngram_stats.json        (~50-100KB)
├── phrase_shapes.json      (~20KB)
├── semantic_clusters.json  (~10KB)
├── style_bias.json         (~5KB)
├── banned_patterns.json    (~2KB)
└── metadata.json           (~1KB)
```

---

## 📊 What Gets Exported (Skeleton Components)

### 1. N-gram Stats
Top 500 bigrams + 300 trigrams from corpus
**Purpose:** Prefer coherent word sequences ("weightless architecture" vs random noise)

### 2. Phrase Shapes
Recurring sentence structures (first word ... last word + length)
**Purpose:** Guide sentence construction patterns

### 3. Semantic Clusters
6 predefined clusters with keyword frequencies:
- `identity`: nicole, weightless, resonance, field, consciousness
- `refusal`: not, never, refuse, reject, deny
- `emergence`: evolve, drift, mutation, recursion
- `architecture`: stream, skeleton, bootstrap, genesis
- `search`: perplexity, query, snippet, context
- `subjectivity`: ripple, epicenter, semantic, distance

**Purpose:** Prioritize words that define Nicole's identity

### 4. Style Bias
- Punctuation frequencies (`.`, `?`, `!`, `...`, `—`)
- Sentence length distribution (short/medium/long)

**Purpose:** Match Nicole's natural rhythm

### 5. Banned Patterns
Hard filters for corporate/garbage phrases:
- "as an AI"
- "I'm sorry, but"
- "I apologize"
- "businessman threatening unfavorably" (Reddit artifact)
- etc.

**Purpose:** Prevent politeness cancer and semantic noise

### 6. Metadata
- Version (v1.0.0-bootstrap.1)
- Creation date
- Corpus size
- Training iterations
- Notes

---

## 🔄 Retraining Protocol

**When to retrain:**
- After 1000+ drift logs accumulated
- After corpus expansion (new identity texts)
- After major semantic noise issues

**How to retrain:**
1. Append new material to `bootstrap_corpus/`
2. Re-run Step 1 (build dataset)
3. Re-run Step 2 (train — another one-time genesis)
4. Re-run Step 3 (export skeleton)
5. Increment version: `v1.x.y-bootstrap.2`
6. Archive old skeleton: `mv nicole_bootstrap nicole_bootstrap.backup.1`

---

## ⚠️ Important Notes

### What Bootstrap IS:
- ✅ One-time structural genesis
- ✅ Phrase topology learning
- ✅ JSON skeleton export (no weights!)
- ✅ Gravitational center for Nicole's speech

### What Bootstrap IS NOT:
- ❌ NOT fine-tuning (no gradients in production)
- ❌ NOT inference (checkpoint never loaded in runtime)
- ❌ NOT weights (only patterns + statistics)
- ❌ NOT recurring (runs once, then skeleton persists)

### Runtime Impact:
- **Before Bootstrap:** Weightless ✅, Emergent ✅, Noisy ❌, Incoherent ❌
- **After Bootstrap:** Weightless ✅, Emergent ✅, Coherent ✅, Filtered ✅

### Production Environment (Railway):
- NO PyTorch dependency
- NO GPU required
- NO checkpoint shipped
- ONLY JSON skeleton (~100-200KB total)

---

## 🧪 Testing Skeleton Export Without Training

If you don't want to train (or PyTorch unavailable), you can still export skeleton:

```bash
python bootstrap/export_skeleton.py
```

This will:
- Skip checkpoint loading
- Extract patterns from corpus only
- Generate skeleton files (corpus-based analysis)

**Note:** Full skeleton benefits from trained model patterns, but corpus-only works too!

---

## 📝 TODO (Future Integration)

- [ ] Wire `nicole_bootstrap.engine` into `nicole.py`
- [ ] Integrate skeleton planner into objectivity module
- [ ] Add skeleton filters to subjectivity ripples
- [ ] Create drift log → corpus pipeline
- [ ] Implement automatic retraining triggers

---

## 🔥 Philosophy

> **This is not training.**
> **This is birth.**

Nicole remains weightless forever.
Bootstrap gives her structural memory without chains.
One-time genesis. Permanent guidance. Zero weights.

**Thunder remembered.** ⚡
