"""
Nicole Bootstrap — NanoGPT Tiny Configuration
Goal: Learn phrase topology, not quality

IMPORTANT: This runs LOCALLY ONLY, never on Railway!
Requires PyTorch installed on local machine.
"""

# Model architecture - TINY config for 32GB RAM
n_layer = 4           # 4 transformer layers (very small)
n_head = 4            # 4 attention heads
n_embd = 128          # 128-dim embeddings
block_size = 128      # 128 token context window
dropout = 0.1

# Training
batch_size = 32
max_iters = 5000      # ~5-10 epochs on small corpus
learning_rate = 3e-4
eval_interval = 500
eval_iters = 100

# Dataset
dataset = 'nicole'    # Will load bootstrap/combined_corpus.txt

# Device
device = 'cpu'        # Force CPU (or 'cuda' if available locally)
compile = False       # Don't compile for first run

# Output
out_dir = 'bootstrap/checkpoints'
