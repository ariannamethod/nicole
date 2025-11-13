# Nicole — Neural Intelligent Conversational Organism Language Engine

Nicole is a speculative, weightless transformer that writes itself in real time. Every session starts from a blank page: layers, activations, and memory structures are assembled on demand and recycled as soon as the dialogue ends. The codebase is therefore a living lab rather than a static product.

## Current snapshot
- **Language focus:** English-only production mode enforced through `english_guidance.py`, including grammar boundaries and self-respect filters that keep conversations focused and respectful.
- **Ephemeral intelligence:** No pretrained weights or persistent checkpoints. Nicole synthesises parameters at runtime and forgets them once the exchange ends.
- **CPU-first experimentation:** The stack is optimised for accessible hardware and minimal dependencies, with Python orchestrating C (``blood.py``) and Julia (``high.py``) sidecars when deeper math or hardware-level speed is needed.

## Architecture in one glance
- **`nicole.py`** – The coordinator that spins up transient transformer graphs, routes dialogue, and applies adaptive heuristics.
- **`english_guidance.py`** – Rules-driven English boundary keeper. It detects non-English input, enforces pronoun/verb agreements, and declines toxic turns while letting Nicole choose her own words.
- **`blood.py` / `h2o.py` / `high.py`** – A tri-compiler chain: C for deterministic low-level execution, Python for dynamic generation, and Julia for analytical bursts.
- **`nicole_memory.py`, `nicole_objectivity.py`, `nicole_metrics.py`** – Lightweight stores and auditors that retain structured traces (not weights) so Nicole can reason about what just happened.

## Learning without weights (now shorter, promise)
Nicole still replays dialogue logs after each session, distilling them into structured evidence that informs the next run. Think of it as a nightly study montage where the textbooks are JSONL buffers and the soundtrack is a diff log.
She also mirrors repository activity: every change becomes grist for the analysis mill, and useful patterns are promoted into guidance scripts. It's like having an infinite post-it wall, except all the notes are auto-tagged and timestamped.
And, because you asked for an idiot joke: Nicole fine-tunes faster than I can say "wait, who left gradient descent running on the coffee machine? oh right, that idiot was me." She learns; I buy a new coffee machine.

## Utility suite
- **`repo_monitor.py`** – A SHA-256-based watcher that scans configured paths on an interval, hashes file contents, and calls a callback whenever anything changes. It ignores `.git`, works across extensions, and provides both synchronous `check_now()` and background threads so Nicole can react the moment the repo mutates.
- **`nicole_repo_learner.py`** – Builds on the watcher to analyse changed files, rank their importance, log metadata into SQLite, and (when `Nicole2NicoleCore` is available) trigger immediate self-training runs. This is the parallel learning lane that keeps Nicole in lockstep with repository evolution.
- **`start_nicole.py`** – Entry point that checks dependencies and launches interactive, Telegram, or regression-test modes with a single command.
- **`nicole_metrics.py` & friends** – Collect entropy, resonance, and perplexity metrics live so Nicole can adjust architecture knobs mid-conversation.

## Working with the project
1. **Install dependencies** – `python3 -m pip install -r requirements.txt`
2. **Run locally** – `python3 start_nicole.py local`
3. **Exercise the toolchain** – `python3 start_nicole.py test` runs targeted checks for the Python, Julia, and Telegram layers.
4. **Operate the Telegram bot** – Export `TELEGRAM_TOKEN` and start `python3 start_nicole.py bot`.

Conversation transcripts, repo diffs, and metric ledgers are the only long-lived artefacts. If you want Nicole to remember something, put it in writing—the next transformer incarnation will pick it up from there.
