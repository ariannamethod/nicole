# Nicole — Neural Intelligent Conversational Organism Language Engine

Nicole is a weightless transformer that assembles itself fresh for every conversation. No checkpoints, no stale heuristics –
just code that reshapes layers, attention, and memory in response to the present dialogue. The current production snapshot runs
English-only, with the language guardrails defined in `english_guidance.py`, while every other subsystem stays ready for
experiments on the next reboot.
---

## Table of contents
1. [Core principles](#core-principles)
2. [Architecture panorama](#architecture-panorama)
3. [Conversation lifecycle](#conversation-lifecycle)
4. [Compiler triad](#compiler-triad)
5. [Operational substrate (AMLK)](#operational-substrate-amlk)
6. [Module reference](#module-reference)
7. [Language guardrails](#language-guardrails)
8. [Memory, metrics, and objectivity](#memory-metrics-and-objectivity)
9. [Repo-coupled evolution](#repo-coupled-evolution)
10. [Self-training overview (short edition)](#self-training-overview-short-edition)
11. [Operational runbook](#operational-runbook)
12. [Developer workflow](#developer-workflow)
13. [Glossary of resonance terminology](#glossary-of-resonance-terminology)
---

## Core principles
- **Weightless intelligence.** Parameters are synthesised on demand and erased after the exchange. Conversation logs and metrics
  are the only persistent traces.
- **English-only boundary.** `english_guidance.py` intercepts non-English input, enforces grammar rules, and declines toxic
  turns without forcing templated phrasing.
- **Tri-compiler architecture.** Python orchestrates, `blood.py` (C) anchors deterministic execution, and `high.py` (Julia)
  delivers analytical bursts when maths need extra acceleration.
- **Repo-coupled evolution.** Tooling watches repository changes and replays them through Nicole’s learning lanes so every
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
- `nicole_metrics.py` doubles as the live telemetry bus that feeds both humans and learners.

### Operational surface
- `start_nicole.py` is the main entry point with `local`, `bot`, and `test` modes.
- `nicole_telegram.py` bridges Nicole into Telegram channels.
- `test_quick_wins.py` exercises critical behaviours without spinning the whole stack.

---

## Conversation lifecycle
Nicole’s runtime can be viewed as a six-act play. Each act corresponds to a concrete piece of code in the repository.

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

---

## Compiler triad
The tri-compiler strategy allows Nicole to manifest cognition across multiple execution domains.

### Blood compiler (`blood.py`)
- Derived from Clang, reduced to deterministic essentials. Every compiled snippet interacts with RAM via explicit pointers.
- Emits cache-local binaries and predictable branch behaviour, making it the low-frequency backbone for hardware proximity.
- Focus areas:
  - Tensor algebra primitives that would be too sluggish in Python.
  - Memory hygiene routines that keep ephemeral tensors from leaking past a session.
  - Deterministic PRNG sequences to match transcripts against reruns.

### H2O bootstrap (`h2o.py`)
- Python bootstrap compiler that generates modules during runtime.
- Enables Nicole to mutate structure mid-conversation without rebooting the stack.
- Supports hotpatching heuristics, injecting new prompt routers, or experimenting with alternative decoding strategies.

### High compiler (`high.py`)
- Julia-based analytical cortex dedicated to statistical routines and symbolic reasoning.
- Exchange tensors with both Python and C pathways, ensuring that mathematics stay fast without losing transparency.
- Typical workloads include resonance matrix updates, topology searches, and higher-order optimisation passes.

---

## Operational substrate (AMLK)
The Arianna Method Linux Kernel lives under `AMLK/` and provides a deterministic launchpad.

- Distilled from Alpine sources, focusing on reproducible boot sequences and stable ABIs.
- OverlayFS, namespaces, and cgroups construct a minimal yet rigorous sandbox for compiler experiments.
- Nicole expects predictable addresses between runs so that cross-language pointers remain trustworthy.
- Consult `AMLK/readme.md` for kernel build instructions, bootstrap scripts, and the philosophy behind the deterministic
  approach.

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
  - Keeps Nicole’s self-modifications evidence-backed and replayable.
- **Key components**
  - `ObjectivityWindow` captures transcript segments and correlates them with metric shifts.
  - `HypothesisLedger` stores research notes, ready for repo learner ingestion.

### `nicole_repo_learner.py`
- **Mission**
  - Bridges repository changes with Nicole’s adaptive heuristics.
  - Parses diffs, ranks their significance, logs outcomes, and can trigger Nicole-to-Nicole sessions.
- **Data flow**
  1. `repo_monitor.py` emits change events.
  2. Learner analyses diff metadata, referencing component registries to see which modules were touched.
  3. SQLite ledger stores findings with timestamps, change fingerprints, and optional follow-up tasks.
  4. Optional training sessions instantiate ephemeral Nicole clones for rehearsal.
- **Extensibility**
  - Custom rankers can be registered to prioritise certain file types (e.g., compilers vs. utilities).
  - Hooks exist for dispatching notifications to dashboards or messaging channels.

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
- **Bridge** between Telegram chat and Nicole’s conversational loop.
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

## Language guardrails
Keeping Nicole English-only is a philosophical and technical constraint.

- **Script detection** rejects non-Latin input early, maintaining focus on the current research domain.
- **Grammar lattice** enforces subject-verb agreement, pronoun sanity, and respectful self-reference.
- **Toxicity filters** decline prompts that would derail experimentation or contradict the project’s ethos.
- **Transparency** – Every refusal includes a brief explanation so logs remain interpretable during audits.
- **Experimentation** – Researchers can add experimental validators but should document them in `english_guidance.py` to keep
  the guardrail map public.

---

## Memory, metrics, and objectivity
These modules form Nicole’s introspective toolkit.

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

### Repo learner internals
- **SQLite schema** captures file paths, diff summaries, metric deltas, and follow-up actions.
- **Nicole-to-Nicole integration** spawns ephemeral instances that rehearse the change; transcripts flow back into the ledger.
- **Human-in-the-loop hooks** allow operators to annotate significant diffs, creating curated knowledge for future runs.

### Repo monitor deep dive
- **Scanning algorithm**
  - Walk directories in deterministic order.
  - Hash file contents with SHA-256.
  - Compare against the previous snapshot stored in memory or disk cache.
  - Produce `added`, `removed`, and `modified` buckets.
- **Threaded watcher**
  - Uses `threading.Thread` with a cooperative stop flag, avoiding daemon surprises.
  - Sleep interval is configurable; defaults favour responsiveness without starving CPU.
- **Callback contract**
  - Receives a payload: `{"added": [...], "modified": [...], "removed": [...], "errors": [...]}`.
  - Each entry includes `path`, `hash_before`, `hash_after`, `timestamp`, and optional diff metadata.
- **Testing strategy**
  - Use the synchronous `check_now()` in unit tests to avoid race conditions.
  - Provide fake callbacks that assert payload structure and sequence.

### Operational advice
- For large monorepos, shard monitors per directory and aggregate results in the learner.
- Store monitor snapshots if you need to rewind to historical baselines.
- Pair monitor alerts with `nicole_metrics.py` to correlate code shifts with behavioural changes.

---

## Self-training overview (short edition)
Nicole still studies after the conversation ends: dialogue logs, metric traces, and repo diffs are distilled into guidance tables
that inform the next run. Think of it as an overnight cram session where the textbooks are JSONL buffers and the study partner is
an SQLite ledger that refuses to sleep.

The learner and monitor cooperate to rank what changed, replay transcripts, and surface the insights Nicole should keep. By the
time a new session begins, the heuristics are refreshed, yet no dense weights were ever stored.

Idiot joke quota: Nicole finishes these self-cram sessions so quickly that the coffee machine barely warms up; I’m still stuck
explaining to facilities why the espresso tastes like gradient clipping.

---

## Operational runbook
Real-world usage often demands structured procedures. This runbook offers playbooks for common scenarios.

### Launching a local research session
1. `python3 -m pip install -r requirements.txt`
2. `python3 start_nicole.py local`
3. Watch the CLI: metrics stream on the side, guardrail notices appear inline when policy triggers.
4. When done, archive the session log for repo learner ingestion.

### Deploying the Telegram bot
1. Export `TELEGRAM_TOKEN`.
2. `python3 start_nicole.py bot`
3. Monitor `nicole_metrics.py` output; configure alerts for resonance spikes.
4. Periodically run `RepoMonitor.check_now()` to keep the learner fed between commits.

### Running regression checks
1. Ensure Julia runtime is installed (see `requirements.txt` for hints).
2. `python3 start_nicole.py test`
3. Inspect `test_quick_wins.py` results; failures often signal guardrail regressions or compiler drift.

### Investigating a behavioural anomaly
1. Consult `nicole_metrics.py` logs for resonance or entropy deviations.
2. Cross-reference repo learner entries to see if recent commits touched relevant modules.
3. Reproduce the conversation using the logs stored in the ledger.
4. Patch modules, commit the fix, and let `repo_monitor.py` broadcast the change for immediate rehearsal.

### Adding a new experimental module
1. Create the module and expose a clear interface.
2. Register it with the repo learner so diffs are classified correctly.
3. Update this README’s Module reference section to keep documentation synced.
4. Add quick-win tests or augment `test_quick_wins.py`.

---

## Developer workflow
Nicole thrives when experiments are documented and reproducible.

- **Environment setup** – Use Python 3.11+, ensure Julia is on the PATH, and compile the `blood.py` toolchain if running low-level
  experiments.
- **Coding style** – Prioritise clarity; avoid magical constants and document heuristics inline.
- **Testing** – Run `python3 start_nicole.py test` after touching orchestration or guardrail logic. For doc-only changes, reference
  commit messages instead.
- **Logging** – When adding new metrics or audit events, ensure they include timestamps and human-readable summaries.
- **Documentation** – Keep sections synchronised with actual code structure. Nicole’s credibility depends on transparent docs.
- **Repo monitor etiquette** – Exclude temporary directories and build artefacts to keep scans efficient.

---

## Glossary of resonance terminology
Nicole’s research vocabulary can feel esoteric; this glossary keeps experiments decipherable.

- **Resonance** – A measure of how harmoniously Nicole’s responses align with previous context without becoming predictable.
- **Entropy** – Statistical dispersion of token probabilities; spikes signal exploration, drops indicate converging certainty.
- **Objectivity window** – A sampled slice of conversation correlated with metric deltas, used for audits.
- **Guidance table** – Lightweight artefact storing distilled heuristics from repo learner analysis.
- **Nicole-to-Nicole** – Ephemeral rehearsal sessions where Nicole trains against a clone to validate new heuristics.
- **Weightless intelligence** – Design principle where no pretrained weights persist between sessions.
- **Arianna Method Linux Kernel (AMLK)** – Deterministic kernel providing a reproducible substrate for Nicole’s compilers.
- **Repo monitor** – SHA-256-based watcher translating filesystem drift into structured events.
- **Learner ledger** – SQLite database capturing repository diffs, metric correlations, and follow-up actions.
- **Resonance anomaly** – Situation where Nicole’s responses deviate sharply from historical patterns, triggering audits.

---

## Running Nicole
For quick reference:

1. **Install dependencies** – `python3 -m pip install -r requirements.txt`
2. **Local interactive run** – `python3 start_nicole.py local`
3. **Regression checks** – `python3 start_nicole.py test`
4. **Telegram bot** – Export `TELEGRAM_TOKEN` then `python3 start_nicole.py bot`

Logs, repo analytics, and metric ledgers are the only long-lived artefacts. If you want Nicole to remember a discovery, write it
into the repository: the next transformer incarnation will study it on boot.

---

## Scenario walkthrough: repo monitor in action
The following narrative demonstrates how Nicole reacts to a concrete commit. Use it as a mental simulator when wiring new
workflows.

1. **Developer edits `english_guidance.py`.**
   - Repo monitor notices the modified file, hashes it, and stores the new fingerprint.
   - Callback payload includes the path, previous hash, new hash, and timestamp.
2. **Learner receives the event.**
   - Checks module registry and classifies the change as *guardrail-critical*.
   - Inserts a row into the SQLite ledger with `priority = HIGH` and attaches the diff summary.
3. **Nicole-to-Nicole rehearsal triggers.**
   - A clone session loads the new guardrail rules and replays recent transcripts that previously failed boundary checks.
   - Metrics are compared against historical baselines to ensure regression coverage.
4. **Objectivity audit logs the rehearsal.**
   - `nicole_objectivity.py` stores the before/after comparison and links it to the commit hash.
   - Operators receive a notification summarising the outcome.
5. **Primary Nicole instance bootstraps later.**
   - During `bootstrap_session()`, Nicole fetches learner hints noting the guardrail update.
   - The new heuristics slide seamlessly into the transformer genesis.
6. **Conversation occurs.**
   - When a user tests the boundary, Nicole references the refreshed rules without missing a beat.
   - Metrics confirm the guardrail now responds faster, reducing latency spikes caused by rule ambiguity.

### Lessons from the walkthrough
- The monitor–learner pipeline transforms raw file edits into actionable behaviour shifts.
- Each subsystem leaves a breadcrumb trail: hashes, ledger entries, rehearsal transcripts, audit snapshots, and metric deltas.
- Because everything is transparent, Nicole can narrate her own upgrades in real time.

---

## Research questions we track next
Nicole is a living laboratory. Here are active investigations documented to keep collaborators aligned.

- **Adaptive grammar heuristics** – Explore probabilistic guardrails that learn from refusal logs without compromising policy.
- **Metric fusion** – Combine resonance and entropy into composite indicators that predict when Nicole should pivot strategies.
- **Repo diff storytelling** – Teach Nicole to explain repository changes conversationally using learner summaries as prompts.
- **Transient compiler specialisation** – Experiment with generating per-conversation C kernels tailored to user intent classes.
- **Human annotation loop** – Integrate optional operator feedback into learner ranking without overriding autonomous discovery.
Contributions tackling these questions should update this list so everyone can see which experiments are live, paused, or retired.
