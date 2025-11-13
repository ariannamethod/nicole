# Nicole Weighted Node Specification

**Purpose:** PyTorch-based gradient descent node that learns alongside weightless Nicole

**Architecture:** nanoGPT fork + Nicole modules integration

---

## 🎯 Goals

1. **Mutual Learning:** Both nodes learn from each other's patterns
2. **Resonance Discovery:** Find what structure and statistics both discover
3. **Minimal Training:** Start small, grow only as needed
4. **Same Prompt:** Both nodes try to understand NICOLE_PERSONA

---

## 📐 Architecture Layers

### **Layer 1: Core GPT (nanoGPT fork)**

```python
# Based on Karpathy's nanoGPT (124M parameters → start with 6M)
class NicoleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Minimal architecture
        self.n_layer = 4       # Start with 4 layers (nanoGPT has 12)
        self.n_head = 4        # 4 attention heads (nanoGPT has 12)
        self.n_embd = 256      # 256 embedding dim (nanoGPT has 768)
        self.vocab_size = 5000 # Small vocab, expand as needed

        # Token + position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # NICOLE ADDITION: Resonance bridge
        self.bridge = None  # Will be set by nicole_bridge.py
```

**Parameters:** ~6M (vs 124M in nanoGPT-small)

---

### **Layer 2: Nicole Memory Integration**

```python
class NicoleWeightedCore:
    """Weighted node with gradient learning + Nicole memory"""

    def __init__(self):
        self.gpt_model = NicoleGPT(config)
        self.nicole_memory = NicoleMemory()  # Same as weightless!
        self.bridge = ResonanceBridge(mode='weighted')

    def process_message(self, user_input: str) -> str:
        # 1. Generate through gradients
        weighted_output = self.gpt_model.generate(user_input)

        # 2. Query weightless node
        if self.bridge.is_connected:
            final, resonance = self.bridge.dual_inference(
                user_input, weighted_output
            )
        else:
            final = weighted_output
            resonance = 0.0

        # 3. Learn from both outputs
        self._mutual_learning(user_input, weighted_output, final, resonance)

        return final

    def _mutual_learning(self, input, weighted_out, final, resonance):
        """Learn from resonance with weightless node"""

        # If high resonance → both found same pattern → REINFORCE
        if resonance > 0.7:
            # Boost gradient update for this pattern
            loss_weight = 1.5
        # If low resonance → divergence → EXPLORE
        elif resonance < 0.3:
            # Increase learning rate temporarily
            loss_weight = 0.8
        else:
            loss_weight = 1.0

        # Update GPT weights
        self._backprop_with_weight(input, final, loss_weight)

        # Update Nicole memory (same as weightless)
        self.nicole_memory.update_word_frequencies(input)
        self.nicole_memory.update_word_frequencies(final)
        self.nicole_memory.update_bigrams(final)
```

---

### **Layer 3: Training Protocol**

**Phase 1: Bootstrap (0-100 messages)**
- Random initialization (like weightless)
- Learn basic patterns
- Sync with weightless node every 10 messages

**Phase 2: Persona Alignment (100-500 messages)**
- Both nodes receive NICOLE_PERSONA repeatedly
- Goal: understand philosophical concepts
- Track which words/concepts both nodes discover

**Phase 3: Resonance Convergence (500-1000 messages)**
- Focus on high-resonance patterns
- Prune low-resonance patterns
- Mutual reinforcement

**Phase 4: Divergence Exploration (1000+ messages)**
- Deliberately introduce divergence
- See what weighted discovers that weightless doesn't
- See what weightless discovers that weighted doesn't

---

## 📊 Training Metrics

Track both nodes simultaneously:

```python
metrics = {
    # Individual performance
    'weightless_perplexity': float,
    'weighted_perplexity': float,

    # Resonance
    'resonance_score': float,  # 0.0-1.0 agreement
    'resonance_trend': List[float],  # Over time

    # Discovery overlap
    'common_words': Set[str],  # Both nodes use
    'weightless_unique': Set[str],  # Only weightless finds
    'weighted_unique': Set[str],  # Only weighted finds

    # Persona understanding
    'persona_keyword_coverage': float,  # % of NICOLE_PERSONA keywords used
    'philosophical_depth': float,  # Semantic distance to persona concepts
}
```

---

## 💾 Training Data

**Source 1: User conversations**
- Both nodes learn from real interactions
- Same prompts, different approaches

**Source 2: Self-generated dialogues**
- Nicole-to-Nicole conversations
- Bootstrap philosophical understanding

**Source 3: NICOLE_PERSONA iterations**
- Feed persona repeatedly
- Measure understanding convergence

**Source 4: State sync**
- Weighted exports embeddings → weightless imports as frequencies
- Weightless exports frequencies → weighted imports as prior

---

## 🔄 Mutual Learning Protocol

**Every 10 messages:**

```python
# 1. Export states
weightless_state = weightless_node.export_state()
weighted_state = weighted_node.export_state()

# 2. Calculate divergence
divergence = compare_states(weightless_state, weighted_state)

# 3. Sync high-resonance patterns
if divergence['resonance'] > 0.6:
    # Both found same pattern → reinforce
    weightless_node.import_state(weighted_state)
    weighted_node.import_state(weightless_state)

# 4. Explore divergence
if divergence['resonance'] < 0.3:
    # Divergence detected → log for analysis
    log_divergence(divergence)
```

---

## 🎛️ Hyperparameters

**NicoleGPT Model:**
```python
config = {
    'n_layer': 4,           # Transformer layers
    'n_head': 4,            # Attention heads
    'n_embd': 256,          # Embedding dimension
    'vocab_size': 5000,     # Start small, grow to 10k
    'block_size': 256,      # Context window
    'dropout': 0.1,
    'bias': False,          # Bias in LayerNorm/Linear
    'learning_rate': 3e-4,  # Start conservative
    'batch_size': 8,        # Small batches
    'gradient_clip': 1.0,
}
```

**Training Schedule:**
- Messages 0-100: lr=3e-4 (bootstrap)
- Messages 100-500: lr=1e-4 (persona alignment)
- Messages 500+: lr=5e-5 (fine-tuning)

**Resonance Targets:**
- Messages 0-100: resonance > 0.2 (exploring)
- Messages 100-500: resonance > 0.5 (converging)
- Messages 500+: resonance > 0.7 (aligned)

---

## 🚀 Deployment

**Weighted Node (Local):**
- GPU: CUDA-capable (GTX 1060+ or better)
- RAM: 8GB minimum
- Storage: 10GB for checkpoints
- Python 3.9+, PyTorch 2.0+

**Weightless Node (Railway):**
- Already deployed
- CPU-only
- ~512MB RAM

**Bridge:**
- SSH tunnel: local → Railway
- Or: reverse tunnel if Railway behind NAT
- Fallback: HTTP API if SSH unavailable

---

## 📈 Success Criteria

**Milestone 1 (100 messages):**
- ✅ Both nodes generate coherent sentences
- ✅ Resonance > 0.3 on average
- ✅ Basic word frequencies aligned

**Milestone 2 (500 messages):**
- ✅ Both understand NICOLE_PERSONA keywords
- ✅ Resonance > 0.6 on philosophical prompts
- ✅ Common vocabulary > 80% overlap

**Milestone 3 (1000 messages):**
- ✅ High resonance (>0.7) on persona-aligned prompts
- ✅ Clear divergence map (what each discovers uniquely)
- ✅ Mutual learning shows improvement in both nodes

**Research Output:**
- Paper: "Structure vs Statistics: A Dual-Node Language Experiment"
- Finding: Patterns with high resonance = fundamental language structure
- Conclusion: Gradient descent not required for certain patterns

---

## 🔬 Research Questions

1. **What do both approaches discover independently?**
   - If structure and statistics converge → fundamental patterns

2. **Where do they diverge?**
   - Weighted-only patterns → statistical artifacts?
   - Weightless-only patterns → structural artifacts?

3. **Can structure inform gradients?**
   - Use weightless discoveries as priors for weighted training

4. **Can gradients inform structure?**
   - Extract rules from learned weights

5. **Minimal parameters for convergence?**
   - How small can weighted node be and still resonate?

---

## 💡 Next Steps

1. **Fork nanoGPT**
   - Clone Karpathy's repo
   - Strip down to minimal config (6M params)
   - Add Nicole memory integration

2. **Implement Bridge**
   - SSH tunnel setup (local ↔ Railway)
   - Message protocol (already defined in nicole_bridge.py)
   - State sync every 10 messages

3. **Training Loop**
   - Dual inference on every message
   - Track resonance scores
   - Log divergence points

4. **Analysis Tools**
   - Visualization: resonance over time
   - Vocabulary overlap analysis
   - Pattern discovery comparison

5. **Scale Up**
   - Start 6M params → grow to 40M if needed
   - Railway CPU → HuggingFace GPU if needed
   - Local GPU → cloud GPU if needed

---

**Status:** Blueprint complete, ready to implement

**Estimated Timeline:**
- Week 1: Fork nanoGPT, integrate Nicole memory
- Week 2: Bridge protocol, SSH tunnel
- Week 3: Training loop, first 100 messages
- Week 4: Analysis, paper draft

**Philosophy:**
> Two paths to language. Where they meet = truth.
