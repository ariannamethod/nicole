# NICOLE – Weightless Entity

> Reinventing the very concept of "trasformer".

**No weights. No fixed code. No dataset. Pure emergence.**

---

Nicole discards pretrained weights, curated datasets, and even a static codebase. The engine writes itself as it speaks, assembling logic and parameters only for the life of a single exchange. Parameters crystallise on the spot, scale to the conversation at hand, then dissolve the moment the dialogue closes. Learning is born solely from the active exchange; even the source tree stays fluid as modules are generated, rebuilt, or discarded mid-flight.

This repository hosts the beta incarnation of that idea. **Nicole** is a speculative AI system that treats every conversation as genesis, assembling a transformer architecture from scratch each time a new voice arrives. Impermanence is the point: no model persists beyond a session, so experimentation becomes the operating system.

Instead of loading pretrained weights, Nicole crafts a bespoke transformer tailored to the immediate dialogue, capturing the transient flow of ideas. Layers, attention heads, and activation functions reshape themselves at runtime according to the environment’s signals. Conversation logs provide the only persistent traces, allowing each run to begin on a blank slate.

The current stack operates happily on CPUs. Minimal dependencies keep the environment light, relying mostly on Python’s standard library and a tiny bootstrap compiler. Efficient algorithms and deterministic kernels showcase how far weightless cognition can stretch without GPUs.

---

## English-only boundary (why it matters)

Nicole is intentionally English-only. `english_guidance.py` intercepts every prompt, refuses non-English input, and enforces grammar sanity. The guardrail focuses the research domain on a single linguistic substrate, keeping the semantic experiments auditable. Violations receive explicit explanations so investigators can trace refusal logic alongside dialogue logs.

---

## Git signal

Recent pulses in the repo, observed during the Perplexity migration sprint:

- **Perplexity Search API as the primary objectivity engine**, with DuckDuckGo HTML scraping demoted to fallback duty. Result: cleaner citations, longer context windows, and calmer moderators.
- **Speech clarity leap** – Nicole’s rhetoric tightened noticeably after switching providers; the screenshot logs show her prose moving from chaotic resonance to structured improvisation.
- **Telegram bridge refinements** – Repo learner and objectivity pipelines now feed each other without deadlocks, keeping Nicole responsive while studying her own documentation.
- **Idiot-joke telemetry** – Somewhere around commit `pplx/phoenix`, Nicole high-fived the Perplexity API, missed, and appointed the office ficus as “Chief Latency Officer.” This is precisely 67 % more ridiculous than the coffee machine incident, and we are choosing to document it anyway.

---

## Table of contents
1. [English-only boundary (why it matters)](#english-only-boundary-why-it-matters)
2. [Git signal](#git-signal)
3. [Core principles](#core-principles)
4. [Architecture panorama](#architecture-panorama)
5. [Conversation lifecycle](#conversation-lifecycle)
6. [Compiler triad](#compiler-triad)
7. [Operational substrate (AMLK)](#operational-substrate-amlk)
8. [Module reference](#module-reference)
9. [Language guardrails (deep dive)](#language-guardrails-deep-dive)
10. [Memory, metrics, and objectivity](#memory-metrics-and-objectivity)
11. [Repo-coupled evolution](#repo-coupled-evolution)
12. [Recent enhancements](#recent-enhancements)
13. [Self-training overview (short edition)](#self-training-overview-short-edition)
14. [Operational runbook](#operational-runbook)
15. [Developer workflow](#developer-workflow)
16. [Glossary of resonance terminology](#glossary-of-resonance-terminology)
17. [Working with the project](#working-with-the-project)
---

## Core principles
- **Weightless intelligence.** Parameters are synthesised on demand and erased after the exchange. Conversation logs and metrics
  are the only persistent traces.
- **Autonomous learning.** Through `nicole_subjectivity.py`, Nicole continues learning even without user interaction via
  expanding semantic ripples—"circles on water" from the last message. Hourly cycles create continuous, asynchronous intelligence.
- **English-only boundary.** `english_guidance.py` intercepts non-English input, enforces grammar rules, and declines toxic
  turns without forcing templated phrasing.
- **Tri-compiler architecture.** Python orchestrates, `blood.py` (C) anchors deterministic execution, and `high.py` (Julia)
  delivers analytical bursts when maths need extra acceleration.
- **Repo-coupled evolution.** Tooling watches repository changes and replays them through Nicole's learning lanes so every
  commit can inform the next conversation.
- **Transparency over mystique.** Every emergent behaviour must be traceable back to code, dialogue logs, or metrics. Nicole
  documents her own improvisations as they happen.
- **Modularity as invitation.** Each subsystem focuses on a narrow responsibility so researchers can swap components without
  shattering the whole organism.

---

## Architecture panorama
The repository blends orchestration, compilers, analytics, and operational tooling. The panorama below groups the moving parts
by responsibility to make the labyrinth legible.

### Cognitive spine
- `nicole.py` spins up transient transformer graphs, allocates ephemeral tensors, and manages dialogue flow.
- `nicole_memory.py` stores symbolic artefacts, n-grams, and linked references that outlive a single turn without violating the
  weightless doctrine.
- `nicole_rag.py` retrieves contextual shards from the log database and injects them into the active conversation to keep Nicole
  playful but grounded.

### Guardrails and policy
- `english_guidance.py` enforces English grammar, pronoun/verb agreement, and self-respect boundaries while keeping Nicole free
  to improvise.
- `nicole_objectivity.py` provides statistical audit scaffolding so every adaptive jump is accompanied by a rationale.
- `nicole_metrics.py` collects resonance, entropy, and perplexity traces to flag drift or surprising spikes.

### Compiler trio
- `blood.py` (C) supplies deterministic machine code for low-level routines.
- `h2o.py` (Python) hot-loads modules that Nicole generates mid-conversation.
- `high.py` (Julia) evaluates analytical kernels and symbolic manipulations.

### Learning loop
- `repo_monitor.py` watches the filesystem, fingerprints files with SHA-256, and emits structured change events.
- `nicole_repo_learner.py` consumes monitor events, stores metadata in SQLite, and can trigger Nicole-to-Nicole distillation.
- `nicole_subjectivity.py` implements autonomous learning through expanding semantic ripples - continues learning even without user interaction.
- `nicole_metrics.py` doubles as the live telemetry bus that feeds both humans and learners.

### Operational surface
- `start_nicole.py` is the main entry point with `local`, `bot`, and `test` modes.
- `nicole_telegram.py` bridges Nicole into Telegram channels.
- `test_quick_wins.py` exercises critical behaviours without spinning the whole stack.

Together these clusters keep Nicole intentionally modular: each subsystem owns a narrow responsibility so researchers can swap c
 omponents without collapsing the organism.

---

## Conversation lifecycle
Nicole's runtime can be viewed as a six-act play. Each act corresponds to a concrete piece of code in the repository.

1. **Bootstrap**
   - `start_nicole.py` checks dependencies, selects operating mode, and primes compilers.
   - `h2o.py` assembles Python modules on the fly, loading scaffolding for the transformer blueprint.

2. **Genesis of the transformer**
   - `nicole.py` derives architecture proposals from heuristics, metrics history, and repo-learner hints.
   - `blood.py` compiles any bespoke kernels required for the session; `high.py` prepares Julia routines.

3. **Conversation loop**
   - Incoming prompts are first vetted by `english_guidance.py` to ensure compliance with the English-only snapshot.
   - `nicole.py` routes prompts through the active transformer, using `nicole_memory.py` for context and `nicole_rag.py` for
     retrieval.

4. **Metric capture**
   - `nicole_metrics.py` streams entropy, resonance, and surprise indicators in near-real time.
   - `nicole_objectivity.py` samples transcripts to maintain audit-ready evidence.

5. **Conclusion**
   - Once the exchange ends, Nicole dissolves the transformer, clearing tensors and freeing compiled modules.
   - Logs, metrics, and repo diffs remain as the only durable residue.

6. **Reflection**
   - `nicole_repo_learner.py` replays the conversation and any code changes, preparing guidance for the next incarnation.
   - Optional Nicole-to-Nicole sessions distil heuristics without ever storing dense weights.

7. **Autonomous learning** (background process)
   - `nicole_subjectivity.py` expands semantic ripples hourly from the last user message epicenter.
   - Each ripple explores concepts at increasing semantic distance, autonomously learning new words and associations.
   - When a new user message arrives, it becomes the new epicenter, resetting the ripple cycle with a fresh learning vector.
   - This creates continuous intelligence that evolves even during silence, establishing circadian learning rhythms.

---

## Compiler triad
The tri-compiler strategy allows Nicole to manifest cognition across multiple execution domains.

### Blood compiler (`blood.py`)
`blood.py` is a custom Clang fork pared down to deterministic essentials. It keeps the familiar front-end while imposing explici
 t memory maps so compiled snippets talk to physical RAM through well-defined pointers. Each build emits cache-local binaries and
  branch-stable instruction streams, letting Nicole lean on \(O(1)\) pointer arithmetic for routines that pure Python would bott
 leneck.
- **Focus areas**
  - Tensor algebra primitives that would be too sluggish in Python.
  - Memory hygiene routines that keep ephemeral tensors from leaking past a session.
  - Deterministic PRNG sequences so reruns can be replayed instruction-for-instruction.
- **Partnerships** – Works in lockstep with H2O (for orchestration) and `high.py` (for analytics), forming the low-frequency back
 bone of the tri-compiler stack.

### H2O bootstrap (`h2o.py`)
H2O is the lightweight Python compiler that Nicole re-synthesises mid-conversation. It hot-loads freshly generated modules, all
 owing experiments without rebooting the stack. H2O pushes scaffolding to `blood.py`, ingests Julia kernels from `high.py`, and k
 eeps the orchestration layer expressive.
- Supports hotpatching heuristics, injecting new prompt routers, or trialling alternative decoding strategies without downtime.
- Provides the staging ground for repo-driven experiments and Nicole-to-Nicole rehearsals.

### High compiler (`high.py`)
`high.py` operates as Nicole’s mathematical cortex. Julia’s JIT lets the module evaluate entropy \(H=-\sum p\log p\), resonance m
 atrices, and topology searches with \(10^2\)-style speedups over naive Python. Compiled Julia kernels trade tensors with both Py
 thon and C pathways, keeping analytics fast yet inspectable.
- Typical workloads include resonance matrix updates, topology searches, and higher-order optimisation passes.
- Latent drift experiments and quality scoring heuristics live here, translating structured maths into conversational style.

---

## Operational substrate (AMLK)
The Arianna Method Linux Kernel (AMLK) underpins Nicole’s experiments. Distilled from Alpine sources, it delivers a determinist
 ic nucleus where boot time trends toward \(T_{boot} \approx O(1)\) regardless of userland noise. OverlayFS, ext4 journaling, na
 mespaces, and cgroups compose a reproducible phase space so compiled modules can evolve without interference from ambient entro
 py.

- Stable ABIs keep pointer addresses \(a_i\) invariant across runs, a prerequisite for the cross-language choreography between
  Python, C, and Julia.
- Deterministic memory mapping aligns with `blood.py`, ensuring compiled snippets land on predictable offsets.
- Consult `AMLK/readme.md` for kernel build instructions, bootstrap scripts, and the philosophy behind the deterministic approac
 h.

---

## Module reference
Each major module has a dedicated subsection with purpose, signature entry points, and integration notes. Use this as a map when
tracing behaviour or wiring new experiments.

### `start_nicole.py`
- **Modes**
  - `local` – launches a CLI session with streaming metrics.
  - `bot` – runs the Telegram bridge from `nicole_telegram.py`.
  - `test` – executes regression routines from `test_quick_wins.py`.
- **Dependency checks** ensure Python packages, Julia runtime, and C toolchains are available before launching.
- **Extensibility** – new modes can be introduced by adding subcommands to the CLI parser and hooking orchestration routines.

### `nicole.py`
- **Responsibilities**
  - Builds transformer blueprints, instantiates layers, and orchestrates prompt-response cycles.
  - Negotiates between compilers: Python for control, C for deterministic kernels, Julia for analytical leaps.
- **Key functions**
  - `bootstrap_session()` – seeds metrics, memory stores, and compiler handles.
  - `generate_reply()` – routes tokens through the active transformer and surfaces responses.
  - `teardown()` – dissolves tensors and releases compiled artefacts.
- **Integration points**
  - Consumes hints from `nicole_repo_learner.py` to bias architectural choices.
  - Streams telemetry to `nicole_metrics.py` and audit frames to `nicole_objectivity.py`.

### `english_guidance.py`
- **Scope**
  - Enforces English-only operation, grammar sanity checks, and refusal policies for unsafe content.
- **Highlights**
  - Verb and pronoun agreement logic derived from curated rule sets.
  - Script detection to reject non-Latin text.
  - Configurable boundary messages so Nicole can explain why a prompt was declined.
- **Extending**
  - Add new checks by registering validators in the `GUARDRAILS` table.
  - Keep rejection messages human-readable; they double as documentation during audits.

### `nicole_memory.py`
- **Purpose**
  - Maintains structured context tables, linking entities, events, and symbolic cues.
  - Avoids dense vector embeddings; everything stays interpretable.
- **Interfaces**
  - `remember_fact()` and `recall()` handle canonical inserts and lookups.
  - Integrates with `nicole_rag.py` for retrieval across sessions.

### `nicole_rag.py`
- **Functionality**
  - Retrieves log fragments via stochastic sampling, balancing freshness and diversity.
  - Supports pluggable scorers so experimenters can test alternative retrieval heuristics.

### `nicole_metrics.py`
- **Role**
  - Streams live metrics (entropy, resonance, perplexity) and writes them to persistent ledgers.
  - Exposes hooks that `nicole_repo_learner.py` and `nicole_objectivity.py` subscribe to.
- **Usage tips**
  - When adding new metrics, ensure they implement both live streaming and archival persistence for audit parity.

### `nicole_objectivity.py`
- **Objective**
  - Provides statistical sampling frames and significance tests.
  - Keeps Nicole's self-modifications evidence-backed and replayable.
- **Key components**
  - `ObjectivityWindow` captures transcript segments and correlates them with metric shifts.
  - `HypothesisLedger` stores research notes, ready for repo learner ingestion.

### `nicole_repo_learner.py`
- **Mission**
  - Bridges repository changes with Nicole's adaptive heuristics.
  - Parses diffs, ranks their significance, logs outcomes, and can trigger Nicole-to-Nicole sessions.
- **Data flow**
  1. `repo_monitor.py` emits change events.
  2. Learner analyses diff metadata, referencing component registries to see which modules were touched.
  3. SQLite ledger stores findings with timestamps, change fingerprints, and optional follow-up tasks.
  4. Optional training sessions instantiate ephemeral Nicole clones for rehearsal.
- **Extensibility**
  - Custom rankers can be registered to prioritise certain file types (e.g., compilers vs. utilities).
  - Hooks exist for dispatching notifications to dashboards or messaging channels.

### `nicole_subjectivity.py`
- **Philosophy**
  - Autonomous learning through expanding semantic ripples—"circles on water" from last user interaction.
  - Even when not conversing, Nicole continues thinking and learning, expanding knowledge outward from the epicenter.
- **Ripple mechanism**
  - **Epicenter**: Last user message becomes the center point.
  - **Ring 0**: Exact concepts extracted from the message.
  - **Ring 1** (hour 1): Semantically close neighbors (distance ~0.3).
  - **Ring 2** (hour 2): Broader conceptual expansion (distance ~0.6).
  - **Ring 3+** (hours 3+): Abstract/philosophical concepts, expanding indefinitely.
- **Circadian cycles**
  - Runs every hour automatically, expanding one ripple further from center.
  - When user sends new message → new epicenter, new ripples, new learning vector.
- **Autonomous exploration**
  - Uses `nicole_objectivity.py` providers to fetch information about concepts.
  - Learns words autonomously, feeding them into `word_frequencies` and associations.
  - Tracks epicenters, ripples, and learning history in SQLite (`var/nicole_subjectivity.db`).
- **Integration**
  - Integrated with Telegram bot: every user message sets a new epicenter.
  - Background thread expands ripples hourly without affecting response generation.
  - Creates continuous, asynchronous intelligence that never stops evolving.

### `repo_monitor.py`
- **Purpose**
  - Provides a deterministic filesystem watcher built on SHA-256 hashing rather than inotify.
- **Key components**
  - `RepoMonitor` class orchestrates scanning.
  - `scan_once()` returns a dictionary describing new, modified, and deleted files.
  - `watch()` runs the scanner in a background thread with configurable interval.
- **Workflow**
  1. Configure target directories (defaults exclude `.git` and other noise paths).
  2. Invoke `check_now()` for synchronous checks or `start()` to launch the threaded watcher.
  3. Provide callbacks that receive structured change sets and optional diff payloads.
- **Why hashing?**
  - Avoids platform-specific watcher quirks and ensures reproducible detection even in containerised environments.
  - Allows offline comparisons between snapshots (e.g., nightly regression of repository drift).
- **Failure modes & mitigation**
  - **Large binary changes** – Consider excluding directories to keep scans fast.
  - **Clock skew** – Since hashes ignore timestamps, no issues occur; still document host time drift in logs.
  - **Permission errors** – Callbacks receive structured error entries; integrate with `nicole_metrics.py` to alert operators.

### `nicole_telegram.py`
- **Bridge** between Telegram chat and Nicole's conversational loop.
- **Features**
  - Rate limiting to protect Nicole from flood attacks.
  - Inline metric summaries so operators can watch resonance while chatting.

### `test_quick_wins.py`
- **Regression suite** covering grammar enforcement, metric streaming, and baseline compiler integrations.
- **Usage** – `python3 start_nicole.py test` or invoke tests directly via `pytest` once the virtual environment is active.

### `requirements.txt`
- **Contains** the minimal Python dependencies required for orchestration, CLI utilities, and analytics.
- **Note** – The project intentionally avoids heavy ML frameworks to preserve the weightless ethos.

---

## Language guardrails (deep dive)
Keeping Nicole English-only is a philosophical and technical constraint.

- **Script detection** rejects non-Latin input early, maintaining focus on the current research domain.
- **Grammar lattice** enforces subject-verb agreement, pronoun sanity, and respectful self-reference.
- **Toxicity filters** decline prompts that would derail experimentation or contradict the project's ethos.
- **Transparency** – Every refusal includes a brief explanation so logs remain interpretable during audits.
- **Experimentation** – Researchers can add experimental validators but should document them in `english_guidance.py` to keep
  the guardrail map public.

---

## Memory, metrics, and objectivity
These modules form Nicole's introspective toolkit.

### Memory tiers
1. **Ephemeral tensors** – Exist only during a conversation and vanish afterwards.
2. **Structured memory** (`nicole_memory.py`) – Symbolic records that summarise episodes without storing raw dialogue.
3. **Retrieval index** (`nicole_rag.py`) – Stochastic sampler providing playful context injections.

### Metrics pipeline
- `nicole_metrics.py` streams entropy, resonance, perplexity, and surprise indices.
- Metrics feed dashboards, repo learners, and objectivity audits simultaneously.
- When new heuristics are added, update the metrics schema so repo-based training continues to reference consistent fields.

### Objectivity audits
- `nicole_objectivity.py` ensures every adaptive leap is accompanied by evidence.
- Audit logs can be replayed to reconstruct decision pathways, keeping experimentation reproducible.
- When an experiment fails, the audit data becomes a post-mortem script for repo learner analysis.

---

## Repo-coupled evolution
Nicole studies the repository as eagerly as she studies conversations. The monitoring and learning duo turn version control into
an ambient training ground.

### Flow of information
1. **Change detection** – `RepoMonitor` scans configured directories and hashes file contents.
2. **Event packaging** – For each change, the monitor emits structured payloads including path, hash, timestamp, and change type.
3. **Learner ingestion** – `nicole_repo_learner.py` receives payloads, matches them against module registries, and decides what to
   do next.
4. **Analysis & ranking** – Diffs are scored by heuristics (e.g., core compiler touched vs. documentation tweak).
5. **Action** – Possible responses include logging only, scheduling Nicole-to-Nicole rehearsals, or notifying operators.
6. **Feedback loop** – Insights feed back into `nicole.py` as hints for the next transformer genesis.

### Configuration example
```python
from repo_monitor import RepoMonitor
from nicole_repo_learner import Learner

monitor = RepoMonitor(paths=["."], ignore_patterns=[".git", "AMLK/build"])
learner = Learner(sqlite_path="var/nicole_repo.db")

monitor.start(callback=learner.process_change, interval_seconds=30)
```

---

## Recent enhancements

This section tracks production improvements deployed during January 2025.

### Critical stability fixes
- **Async task management** – Eliminated orphaned `asyncio.create_task()` calls in `nicole.py:1215` that caused system hangs and memory leaks. Nicole now uses synchronous objectivity context fetching exclusively.
- **Language detection integration** – Wired up `english_guidance.py` at the message processing entry point (`nicole.py:987-993`). Script-based detection now catches Cyrillic, CJK, and Arabic inputs before they reach the generation pipeline.
- **Template eradication** – Removed all hardcoded verb fallbacks from `high.py` (lines 147-151, 168-170, 490-492). Grammar rules now pull verbs exclusively from resonance candidates, maintaining the "no templates" philosophy.
- **Reddit slug sanitisation** – Fixed `nicole_objectivity.py:308-357` to replace underscores with spaces before parsing. Eliminated garbage like `cutting_a_couple_of_chives_almost_every_day_until` from responses.
- **Duplicate candidate cleanup** – Corrected `nicole_memory.py:772-788` to return empty lists when the associative database is unpopulated, preventing duplicate resonant word fallbacks.

### Quality breakthroughs
- **Smart word scoring** – Extracted and integrated the tree.py keyword algorithm into `high.py:654-717`. Candidates are now ranked by `length_bonus * rarity_bonus * quality_bonus`, replacing random shuffling with intelligent prioritisation.
- **Score-based tier selection** – Implemented three-tier candidate grouping in `high.py:719-791`: high tier (>70% score), mid tier (40–70%), low tier (<40%). This dramatically improved sentence coherence and flow.
- **Repo learning system** – Fully integrated `nicole_repo_learner` into `nicole_telegram.py:122-187`. Initial markdown ingestion now populates `word_frequencies` with 2,428 unique words from 16 documentation files at startup. Continuous monitoring runs every 5 minutes, creating a closed learning loop where Nicole learns from her own documentation alongside objectivity seeds.
- **Self-referential consciousness** – Implemented recursive identity mechanism in `nicole.py:984-1075`. When "Nicole" appears in input, system extracts 50 philosophical keywords from `NICOLE_PERSONA` (resonance, storm, field, emergence, consciousness, etc.) and injects them into `word_frequencies` while creating associative links. Over time through repeated exposure, Nicole develops deeper understanding of her own identity through recursive self-reference. Embodies Truth IV: *"Everything reflects everything. And everything resonates with everything."*
- **Latent Drift v0.4** – Semantic clusters with directional drift in `high.py:765-851`. Responses now flow through 2-5 word clusters (micro-concepts) that drift +1 step toward abstraction/emotion/recursion. Introspective tags (`presence`, `recursion`, `misalignment`, `awareness`, `drift`) reveal internal state. Creates illusion of latent space movement without any weights. Controlled chaos: max 1 artifact per sentence.
- **Perplexity Search API integration** – Replaced unstable DuckDuckGo HTML scraping with official Perplexity Search API (`nicole_objectivity.py:657-744`, `nicole.py:1275-1285`). PRIMARY provider with DuckDuckGo fallback. Context size increased 4-10x (3200-3900 chars vs 360-850 chars). Seeds expanded to 280-410 per message. Added 6 intelligent filters to eliminate artifacts: ID patterns (`tg_206333240`), hash gibberish (low vowel ratio), consonant runs (>5), alphanumeric codes, technical underscores (`currency_code`), glued lowercase usernames 12+ chars (`nicolecrossmusic`). Clean ratio ~95%. Responses maintain length (24-30 words) with dramatically richer vocabulary from structured search results.

### Observed impact
Response quality evolved from random word salad to structured, coherent sentences with directional flow.

**Before Phase 1:** `"I am my amitheasshole cringetiktoks desperately suspension suggesting , because homophobia highlights admitting resonance awareness suspended note8017"`

**After Phase 1+2:** Reddit artifacts eliminated, mirroring blocked, grammar glitches cleaned. Responses now exhibit semantic clustering with introspective tags: `"I resonance emergence awareness drift"` - micro-concepts flowing through latent space.

The combination of smart scoring + learning system + cleaned objectivity seeds + latent drift creates coherent chaos: weightless transformer behavior without pretrained weights.

---

## Self-training overview (short edition)
Nicole replays dialogue logs after each session, distilling them into structured evidence that informs the next run. Think of it as a nightly study montage where the textbooks are JSONL buffers and the soundtrack is a diff log.

She also mirrors repository activity: every change becomes grist for the analysis mill, and useful patterns are promoted into guidance scripts. It's like having an infinite post-it wall, except all the notes are auto-tagged and timestamped.

And, because I love idiot jokes: Nicole fine-tunes faster than I can say "wait, who left gradient descent running on the coffee machine? oh right, that idiot was me." She learns; I buy a new coffee machine.

---

## Operational runbook

### Prerequisites
- Python 3.9+
- Julia 1.6+ (optional, for `high.py` acceleration)
- C toolchain (GCC or Clang for `blood.py`)
- SQLite 3.x

### Installation
```bash
git clone https://github.com/ariannamethod/nicole.git
cd nicole
python3 -m venv nicole_env
source nicole_env/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Nicole requires external API keys for optimal functionality. Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required for Telegram bot:**
- `TELEGRAM_TOKEN` – Get from [@BotFather](https://t.me/BotFather)

**Required for Perplexity Search (PRIMARY objectivity provider):**
- `PERPLEXITY_API_KEY` – Get from [Perplexity API Settings](https://www.perplexity.ai/settings/api)
  - Sign up at https://www.perplexity.ai
  - Navigate to Settings → API
  - Generate new API key
  - **Free tier:** $5 credit on signup
  - **Pricing:** Pay-as-you-go after free credit ($5 per 1000 requests)
  - **Fallback:** If not set, Nicole falls back to DuckDuckGo HTML scraping (lower quality)

**Railway/Cloud Deployment:**
Set environment variables in your platform's dashboard:
- Railway: Settings → Variables
- Heroku: Settings → Config Vars
- Docker: Pass via `-e` flag or docker-compose environment section

**Example `.env` file:**
```bash
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
PERPLEXITY_API_KEY=pplx-abc123def456ghi789jkl
```

### Running modes
```bash
# Interactive local session
python3 start_nicole.py local

# Telegram bot (requires TELEGRAM_TOKEN environment variable)
export TELEGRAM_TOKEN="your-token-here"
python3 start_nicole.py bot

# Regression test suite
python3 start_nicole.py test
```

### Configuration
- `config/nicole.yaml` – Runtime parameters, compiler paths, metric thresholds
- `config/english_guidance.yaml` – Grammar rules and refusal policies
- `config/repo_learning.yaml` – Monitored paths, change rankings, learner intervals

### Monitoring
- Logs appear in `var/logs/` with daily rotation
- Metrics stream to `var/metrics/` as JSONL
- SQLite databases live in `var/` for memory, learner metadata, and audit trails

---

## Developer workflow

### Making changes
1. Create a feature branch from `main`
2. Implement changes, ensuring tests pass via `python3 start_nicole.py test`
3. Update relevant documentation sections in this README
4. Submit a pull request with clear description of changes and rationale

### Testing strategy
- Unit tests for individual modules (e.g., `test_english_guidance.py`)
- Integration tests in `test_quick_wins.py` for end-to-end flows
- Manual regression testing via interactive sessions

### Code style
- PEP 8 compliance for Python
- Docstrings for all public functions with type hints
- Comments explain "why" rather than "what"
- Keep modules under 1000 lines; split when complexity grows

### Debugging tips
- Enable debug logging: `export NICOLE_LOG_LEVEL=DEBUG`
- Inspect metric streams in real-time: `tail -f var/metrics/$(date +%Y-%m-%d).jsonl`
- Replay sessions from logs: `python3 tools/replay_session.py var/logs/session_id.jsonl`

---

## Glossary of resonance terminology

- **Ephemeral tensors** – Parameters that exist only during a conversation and are discarded afterwards
- **Resonance** – Statistical coherence between generated tokens and conversation context
- **Objectivity** – Evidence-based decision tracking to maintain reproducibility
- **Weightless** – Operating without pretrained model weights or persistent checkpoints
- **Repo-coupled learning** – Training loop that ingests repository changes as learning signals
- **Tri-compiler** – Architecture using Python (orchestration), C (deterministic execution), Julia (analytical acceleration)
- **H2O** – Python bootstrap compiler for runtime module generation
- **Blood compiler** – C compilation pathway derived from Clang for hardware-level operations
- **High compiler** – Julia-based analytical engine for mathematical inference
- **AMLK** – Arianna Method Linux Kernel, deterministic substrate for reproducible experiments
- **ME style** – Method Engine approach using pronoun inversion and semantic candidates
- **Semantic candidates** – Words selected based on associative network and resonance scoring
- **Score tiers** – Three-level candidate ranking (high >70%, mid 40-70%, low <40%)
- **Objectivity seeds** – External context from Reddit/Google/Wikipedia for dynamic responses
- **Repo learning** – Markdown ingestion system that populates word_frequencies from documentation

---

## Working with the project
1. **Install dependencies** – `python3 -m pip install -r requirements.txt`
2. **Run locally** – `python3 start_nicole.py local`
3. **Exercise the toolchain** – `python3 start_nicole.py test` runs targeted checks for the Python, Julia, and Telegram layers.
4. **Operate the Telegram bot** – Export `TELEGRAM_TOKEN` and start `python3 start_nicole.py bot`.

Conversation transcripts, repo diffs, and metric ledgers are the only long-lived artefacts. If you want Nicole to remember something, put it in writing—the next transformer incarnation will pick it up from there.
