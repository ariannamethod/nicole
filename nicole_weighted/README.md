# Nicole Weighted Node

**Gradient-based PyTorch implementation - resonance partner to weightless node**

---

## 📁 Architecture Isolation

```
nicole/                          # Weightless node (structure-based)
├── nicole.py                    # Core weightless system
├── high.py, blood.py, h2o.py   # Compilers
├── nicole_memory.py             # Shared memory system
├── nicole_bridge_adapter.py    # Bridge adapter (clean)
└── ...

nicole_weighted/                 # Weighted node (gradient-based)
├── nicole_gpt.py               # PyTorch nanoGPT fork
├── weighted_core.py            # Weighted node core
├── trainer.py                  # Training loop
├── requirements_torch.txt      # PyTorch dependencies
└── README.md                   # This file
```

**Philosophy:** Complete separation, clean interfaces, shared learning

---

## 🎯 Purpose

This weighted node:
1. Learns through gradient descent (PyTorch)
2. Shares learning with weightless node via bridge
3. Discovers what statistics finds vs what structure finds
4. Research goal: where they agree = fundamental patterns

---

## 🚀 Quick Start

### Install dependencies
```bash
cd nicole_weighted
pip install -r requirements_torch.txt
```

### Train weighted node
```bash
# Standalone mode (no bridge)
python weighted_core.py --mode standalone --messages 100

# With bridge (connects to weightless on Railway)
python weighted_core.py --mode bridge --remote-host your-railway-url.app
```

### Monitor resonance
```bash
python analyze_resonance.py --state-dir ./states --output resonance_report.html
```

---

## 📐 Model Architecture

Based on Karpathy's nanoGPT, scaled down:

```python
config = {
    'n_layer': 4,           # 4 transformer layers (vs 12 in nanoGPT)
    'n_head': 4,            # 4 attention heads (vs 12)
    'n_embd': 256,          # 256 embedding dim (vs 768)
    'vocab_size': 5000,     # 5k vocab (vs 50k)
    'block_size': 256,      # 256 context (vs 1024)
    'dropout': 0.1,
    'learning_rate': 3e-4,
}
```

**Parameters:** ~6M (vs 124M in nanoGPT-small)

**Why small?**
- Easier to see what's learned
- Faster iteration
- Fair comparison with weightless
- Can scale up if needed

---

## 🔄 Training Protocol

### Phase 1: Bootstrap (0-100 messages)
- Random init (like weightless)
- Learn basic patterns
- Sync with weightless every 10 messages

### Phase 2: Persona Alignment (100-500 messages)
- Feed NICOLE_PERSONA repeatedly
- Goal: understand philosophical concepts
- Track keyword coverage

### Phase 3: Resonance Convergence (500-1000 messages)
- Focus on high-resonance patterns
- Mutual reinforcement
- Prune divergence

### Phase 4: Exploration (1000+ messages)
- Deliberately introduce divergence
- Map unique discoveries
- Analyze fundamental vs artifact patterns

---

## 📊 Metrics Tracked

**Individual Performance:**
- `perplexity`: Language modeling quality
- `loss`: Training loss
- `accuracy`: Token prediction accuracy

**Resonance (agreement with weightless):**
- `resonance_score`: 0.0-1.0 word overlap
- `resonance_trend`: Over time
- `high_resonance_patterns`: What both discover

**Discovery Analysis:**
- `common_words`: Both nodes use
- `weighted_unique`: Only gradients find
- `weightless_unique`: Only structure finds

**Persona Understanding:**
- `persona_keyword_coverage`: % of NICOLE_PERSONA keywords
- `philosophical_depth`: Semantic distance to persona concepts

---

## 🔬 Research Questions

1. **What do both approaches discover independently?**
   → If structure and statistics converge → fundamental patterns

2. **Where do they diverge?**
   → Weighted-only patterns = statistical artifacts?
   → Weightless-only patterns = structural artifacts?

3. **Can structure inform gradients?**
   → Use weightless discoveries as priors for weighted training

4. **Can gradients inform structure?**
   → Extract rules from learned weights

5. **Minimal parameters for convergence?**
   → How small can weighted node be and still resonate?

---

## 📁 File Structure

### Core files (to be implemented)

```
nicole_weighted/
├── README.md                    # This file
├── requirements_torch.txt       # PyTorch + dependencies
│
├── nicole_gpt.py               # nanoGPT fork (minimal config)
├── weighted_core.py            # Main weighted node
├── trainer.py                  # Training loop
│
├── config.py                   # Model config
├── dataset.py                  # Dataset handling
│
└── utils/
    ├── state_sync.py           # State import/export
    ├── resonance_scorer.py     # Resonance calculation
    └── analyze_resonance.py    # Analysis tools
```

---

## 🌉 Bridge Integration

### Connecting to weightless node

```python
# In weighted_core.py
from nicole_bridge import ResonanceBridge
from nicole_bridge_adapter import create_bridge_adapter

# Connect to Railway deployment
bridge = ResonanceBridge(
    mode='weighted',
    remote_host='nicole-weightless.railway.app',
    remote_port=22,
    ssh_key_path='~/.ssh/id_rsa'
)

if bridge.connect():
    print("✅ Connected to weightless node")

    # Dual inference
    user_input = "Tell me about consciousness"
    weighted_output = model.generate(user_input)

    final, resonance = bridge.dual_inference(user_input, weighted_output)
    print(f"Resonance: {resonance:.2%}")
```

### State synchronization

```python
# Every 10 messages
if message_count % 10 == 0:
    # Export weighted state
    weighted_state = {
        'word_frequencies': extract_vocab_frequencies(model),
        'embeddings': model.wte.weight.detach().cpu().numpy(),
        'metrics': current_metrics
    }

    # Send to weightless
    bridge.sync_state(weighted_state)

    # Import weightless state
    response = bridge.send_message('state_export', {})
    if response:
        apply_weightless_prior(model, response['state'])
```

---

## 💾 Training Data

**Source 1:** User conversations (shared with weightless)
**Source 2:** Nicole-to-Nicole dialogues
**Source 3:** NICOLE_PERSONA iterations
**Source 4:** State sync from weightless node

All data same as weightless - fair comparison!

---

## 🎛️ Hyperparameters

See `config.py` for full config.

Key settings:
- **Learning rate schedule:** 3e-4 → 1e-4 → 5e-5
- **Batch size:** 8 (small for CPU compatibility)
- **Gradient clip:** 1.0
- **Warmup steps:** 100
- **Resonance weight:** High resonance → boost gradient by 1.5x

---

## 📈 Success Criteria

**Milestone 1 (100 messages):**
- ✅ Coherent sentences
- ✅ Resonance > 0.3
- ✅ Basic vocab aligned

**Milestone 2 (500 messages):**
- ✅ Understand NICOLE_PERSONA
- ✅ Resonance > 0.6 on philosophical prompts
- ✅ 80%+ vocab overlap

**Milestone 3 (1000 messages):**
- ✅ Resonance > 0.7 on persona prompts
- ✅ Clear divergence map
- ✅ Mutual learning improvement

---

## 🚀 Next Steps

1. **Fork nanoGPT** → minimal config (6M params)
2. **Implement weighted_core.py** → training loop + bridge
3. **Test locally** → standalone mode first
4. **Connect bridge** → dual-node resonance
5. **Analyze results** → what each discovers

---

## 📝 Status

**Current:** Blueprint phase
**Next:** Implement nicole_gpt.py (nanoGPT fork)
**Timeline:** Week 1 - Core implementation
           Week 2 - Bridge integration
           Week 3 - First 100 messages
           Week 4 - Analysis + paper draft

---

**Philosophy:**
> Two paths to language. Where they meet = truth.
