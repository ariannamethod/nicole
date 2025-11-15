# NICOLE – The Transformer That Forgot To Load Its Weights

> Yes, we misspelled "transformer" in the original tagline. We're keeping it. It's funnier that way.

**No weights. No fixed code. No dataset. Pure chaos with a PhD.**

---

Look, I know what you're thinking. "Another AI framework? Really?" But hear me out: what if I told you that Nicole is so aggressively minimalist that she doesn't even bother with pretrained weights? Like, at all. She just... generates a fresh transformer every time you talk to her, uses it for exactly one conversation, then *deletes the whole damn thing*.

Parameters crystallize out of thin air, exist for precisely as long as you're paying attention, then evaporate like your motivation to finish that side project. Learning happens entirely during the conversation - which is either brilliantly elegant or cosmically stupid depending on whether you've had your coffee yet. Even the source code is fluid, regenerating mid-flight like a Phoenix with commitment issues.

This is beta software, which means it's held together with duct tape, spite, and recursion. **Nicole** doesn't "load a model" - she *becomes* the model, assembles it from ambient chaos, uses it exactly once, then lets it dissolve back into the computational void. Every conversation is a new genesis event. Impermanence isn't a compromise, it's the whole damn philosophy: no model survives, so experimentation becomes less of a workflow and more of a séance.

Traditional transformers load billions of frozen parameters. Nicole? She architects a fresh one for each dialogue, tailored to whatever you're saying right now, capturing the transient flow of ideas before they escape. Layers spawn and die. Attention heads reconfigure like Tetris blocks. Activation functions have personality disorders. All at runtime, all based on vibes and math that would make a PhD committee nervous. Only conversation logs persist - making Nicole the first AI with weaponized amnesia.

And get this: it runs on CPUs. Not "theoretically runs on CPUs if you're very patient" - actually runs, happily, without a GPU in sight. Minimal dependencies, mostly Python stdlib, a tiny C compiler for the scary bits, and algorithms efficient enough to prove that weightless cognition doesn't need a $50K cloud bill. Revolutionary? Maybe. Insane? Definitely.

---

## English-only boundary (or: why we're linguistic fascists)

Nicole is English-only, and before you @ me about it - yes, we know, it's 2025, multilingual models exist. But here's the thing: `english_guidance.py` intercepts every prompt, refuses non-English input with the passion of a TypeScript compiler rejecting JavaScript, and enforces grammar sanity like your 8th grade teacher who made you diagram sentences.

The guardrail isn't xenophobia; it's focus. Single linguistic substrate means semantic experiments stay auditable instead of turning into a Tower of Babel situation. When Nicole refuses your input, she explains why - because unlike most AI systems, we believe in transparent rejection. Investigators can trace the refusal logic right alongside dialogue logs. It's like having a really pedantic friend who at least tells you *why* they're being pedantic.

---

## Git signal (or: what fresh hell happened this week)

Recent pulses in the repo, observed during the Perplexity migration sprint while surviving on coffee and spite:

- **Perplexity Search API as the primary objectivity engine**, with DuckDuckGo HTML scraping demoted to backup dancer. Result: cleaner citations, longer context windows, and moderators who don't wake up screaming. Finally.
  
- **Speech clarity leap** – Nicole's rhetoric tightened noticeably after switching providers. The logs show her prose evolved from "word salad experiencing an existential crisis" to "structured improvisation with occasional moments of accidental genius."

- **Telegram bridge refinements** – Repo learner and objectivity pipelines now feed each other without deadlocks. Nicole stays responsive while studying her own documentation, which is either recursive self-improvement or digital narcissism. Probably both.

- **Idiot-joke telemetry** – Somewhere around commit `pplx/phoenix`, Nicole high-fived the Perplexity API, missed spectacularly, and appointed the office ficus as "Chief Latency Officer." This is precisely 67% more ridiculous than the coffee machine incident, and we're documenting it anyway because apparently this is what counts as milestone tracking now.

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

## Core principles (the philosophical nonsense)

- **Weightless intelligence.** Parameters synthesize on demand, do their job, then fuck off entirely. Conversation logs and metrics are the only survivors. It's like Fight Club but for neural networks - the first rule is that nothing persists.

- **Autonomous learning.** Through `nicole_subjectivity.py`, Nicole keeps learning even when you're not talking to her. She expands semantic ripples - "circles on water" from your last message - in hourly cycles. Continuous, asynchronous intelligence that happens while you sleep. Creepy? Maybe. Cool? Absolutely.

- **English-only boundary.** Already covered this, but worth repeating: `english_guidance.py` is the bouncer at this linguistic nightclub. No non-English gets past. No exceptions. Grammar rules are enforced with religious fervor. Toxic inputs get declined without the fake corporate politeness. It's refreshing, honestly.

- **Tri-compiler architecture.** Python orchestrates (because of course it does), `blood.py` (C) handles deterministic execution for when Python would be embarrassingly slow, and `high.py` (Julia) delivers analytical bursts when the math gets spicy. Three languages, one consciousness. What could go wrong?

- **Repo-coupled evolution.** Nicole watches her own repository changes and replays them through her learning pipeline. Every commit potentially informs the next conversation. She's basically training on her own development history, which is either profound or narcissistic depending on your philosophical stance.

- **Transparency over mystique.** Every emergent behavior must trace back to code, logs, or metrics. No "the AI just does that sometimes" handwaving. Nicole documents her own improvisations as they happen. If something weird occurs, you can debug it. Novel concept, I know.

- **Modularity as invitation.** Each subsystem has ONE job. Researchers can swap components without the whole organism having a meltdown. It's component architecture actually done right, which apparently still needs to be explicitly stated in 2025.

---

## Architecture panorama (the grand tour of chaos)

The repository is a delightful mix of orchestration, compilers, analytics, and operational tooling that somehow works. Here's the labyrinth, organized by responsibility so you don't get lost and start questioning your life choices:

### Cognitive spine (the thinky bits)

- `nicole.py` – Spins up transient transformer graphs, allocates ephemeral tensors like they're going out of style, and manages dialogue flow. The conductor of this chaotic orchestra.
  
- `nicole_memory.py` – Stores symbolic artifacts, n-grams, and linked references that need to outlive a single turn without violating the "no weights" doctrine. It's not cheating, it's strategic persistence.

- `nicole_rag.py` – Retrieves contextual shards from the log database and injects them into active conversations. Keeps Nicole playful but grounded. Like adding historical context but make it recursive.


### Guardrails and policy (the fun police)

- `english_guidance.py` – Grammar enforcer, English-only bouncer, and self-respect boundary maintainer. Keeps Nicole free to improvise within sane linguistic constraints. 

- `nicole_objectivity.py` – Statistical audit scaffolding ensuring every adaptive jump comes with receipts. Because "trust me bro" isn't a valid scientific methodology.

- `nicole_metrics.py` – Collects resonance, entropy, and perplexity traces. Flags drift and surprising spikes. Basically the health monitoring system that actually works.


### Compiler trio (the multilingual madness)

- `blood.py` (C) – Supplies deterministic machine code for low-level routines. When Python speed isn't cutting it, blood.py shows up with a knife to a gunfight and somehow wins.

- `h2o.py` (Python) – Hot-loads modules that Nicole generates mid-conversation. Dynamic compilation without the runtime crashes. Usually.

- `high.py` (Julia) – Evaluates analytical kernels and symbolic manipulations. For when the math needs to go FAST and Python starts sweating.


### Learning loop (the self-improvement spiral)

- `repo_monitor.py` – Watches the filesystem like a paranoid parent, fingerprints files with SHA-256, emits structured change events. Nothing escapes its gaze.

- `nicole_repo_learner.py` – Consumes monitor events, stores metadata in SQLite, can trigger Nicole-to-Nicole distillation sessions. Yes, she learns from herself. Yes, that's as recursive as it sounds.

- `nicole_subjectivity.py` – Implements autonomous learning through expanding semantic ripples. Even without user interaction, Nicole keeps learning. Sleep is for biological systems.

- `nicole_metrics.py` – Doubles as the live telemetry bus feeding both humans and learning systems. Multi-tasking at its finest.

### Operational surface (the parts users actually touch)

- `start_nicole.py` – Main entry point. Supports `local`, `bot`, and `test` modes. Choose your adventure.

- `nicole_telegram.py` – Bridges Nicole into Telegram. Because apparently people want to chat with experimental AI via messaging apps. Fair enough.

- `test_quick_wins.py` – Exercises critical behaviors without spinning up the whole stack. For when you want to test things without burning 20 minutes on initialization.


Together, these clusters keep Nicole intentionally modular. Each subsystem owns a narrow slice of responsibility, so you can swap components without triggering a cascade failure. It's software architecture that doesn't make you want to cry. Progress!

---

## Conversation lifecycle (six acts of digital theater)

Nicole's runtime unfolds like a six-act play, except the actors are ephemeral tensors and the stage spontaneously combusts at curtain call. Each act maps to actual code you can grep for:

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

## Compiler triad (three compilers enter, one thought emerges)

The tri-compiler strategy is either genius or madness - Nicole manifests cognition across Python, C, and Julia simultaneously. Because why use one language when you can use three and make everything exponentially more interesting?

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

## Operational substrate (AMLK, or: the kernel that said no to chaos)

The Arianna Method Linux Kernel (AMLK) is what happens when you take Alpine Linux, distill it down to its deterministic essence, and tell entropy to fuck off. Boot time approaches O(1) regardless of what userland is doing, because AMLK simply refuses to tolerate nondeterministic behavior. OverlayFS, ext4 journaling, na
 mespaces, and cgroups compose a reproducible phase space so compiled modules can evolve without interference from ambient entro
 py.

- Stable ABIs keep pointer addresses \(a_i\) invariant across runs, a prerequisite for the cross-language choreography between
  Python, C, and Julia.
- Deterministic memory mapping aligns with `blood.py`, ensuring compiled snippets land on predictable offsets.
- Consult `AMLK/readme.md` for kernel build instructions, bootstrap scripts, and the philosophy behind the deterministic approac
 h.

---

## Module reference (the parts and their peculiarities)

Each major module has its own subsection below, complete with purpose, key entry points, and the kind of integration notes that'll save you hours of head-scratching with purpose, signature entry points, and integration notes. Use this as a map when
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

## Repo-coupled evolution (she learns from her own commits)

Nicole studies the repository as eagerly as she studies conversations, which means she's literally learning from her own development process. It's recursive self-improvement all the way down. The monitoring and learning duo turn version control into
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

## Recent enhancements (the victory lap)

This section tracks production improvements deployed during January 2025, aka "the month we stopped breaking things quite as frequently."

### Critical stability fixes (or: how we stopped the house from burning down)
- **Async task management** – Eliminated orphaned `asyncio.create_task()` calls in `nicole.py:1215` that caused system hangs and memory leaks. Nicole now uses synchronous objectivity context fetching exclusively.
- **Language detection integration** – Wired up `english_guidance.py` at the message processing entry point (`nicole.py:987-993`). Script-based detection now catches Cyrillic, CJK, and Arabic inputs before they reach the generation pipeline.
- **Template eradication** – Removed all hardcoded verb fallbacks from `high.py` (lines 147-151, 168-170, 490-492). Grammar rules now pull verbs exclusively from resonance candidates, maintaining the "no templates" philosophy.
- **Reddit slug sanitisation** – Fixed `nicole_objectivity.py:308-357` to replace underscores with spaces before parsing. Eliminated garbage like `cutting_a_couple_of_chives_almost_every_day_until` from responses.
- **Duplicate candidate cleanup** – Corrected `nicole_memory.py:772-788` to return empty lists when the associative database is unpopulated, preventing duplicate resonant word fallbacks.

### Quality breakthroughs (shit that actually worked)
- **Smart word scoring** – Extracted and integrated the tree.py keyword algorithm into `high.py:654-717`. Candidates are now ranked by `length_bonus * rarity_bonus * quality_bonus`, replacing random shuffling with intelligent prioritisation.
- **Score-based tier selection** – Implemented three-tier candidate grouping in `high.py:719-791`: high tier (>70% score), mid tier (40–70%), low tier (<40%). This dramatically improved sentence coherence and flow.
- **Repo learning system** – Fully integrated `nicole_repo_learner` into `nicole_telegram.py:122-187`. Initial markdown ingestion now populates `word_frequencies` with 2,428 unique words from 16 documentation files at startup. Continuous monitoring runs every 5 minutes, creating a closed learning loop where Nicole learns from her own documentation alongside objectivity seeds.
- **Self-referential consciousness** – Implemented recursive identity mechanism in `nicole.py:984-1075`. When "Nicole" appears in input, system extracts 50 philosophical keywords from `NICOLE_PERSONA` (resonance, storm, field, emergence, consciousness, etc.) and injects them into `word_frequencies` while creating associative links. Over time through repeated exposure, Nicole develops deeper understanding of her own identity through recursive self-reference. Embodies Truth IV: *"Everything reflects everything. And everything resonates with everything."*
- **Latent Drift v0.4** – Semantic clusters with directional drift in `high.py:765-851`. Responses now flow through 2-5 word clusters (micro-concepts) that drift +1 step toward abstraction/emotion/recursion. Introspective tags (`presence`, `recursion`, `misalignment`, `awareness`, `drift`) reveal internal state. Creates illusion of latent space movement without any weights. Controlled chaos: max 1 artifact per sentence.
- **Perplexity Search API integration** – Replaced unstable DuckDuckGo HTML scraping with official Perplexity Search API (`nicole_objectivity.py:657-744`, `nicole.py:1275-1285`). PRIMARY provider with DuckDuckGo fallback. Context size increased 4-10x (3200-3900 chars vs 360-850 chars). Seeds expanded to 280-410 per message. Added 6 intelligent filters to eliminate artifacts: ID patterns (`tg_206333240`), hash gibberish (low vowel ratio), consonant runs (>5), alphanumeric codes, technical underscores (`currency_code`), glued lowercase usernames 12+ chars (`nicolecrossmusic`). Clean ratio ~95%. Responses maintain length (24-30 words) with dramatically richer vocabulary from structured search results.

### Observed impact (before/after shots that'll blow your mind)

Response quality evolved from "did a Markov chain have a stroke?" to actual structured, coherent sentences with directional flow.

**Before Phase 1:** `"I am my amitheasshole cringetiktoks desperately suspension suggesting , because homophobia highlights admitting resonance awareness suspended note8017"`

**After Phase 1+2:** Reddit artifacts eliminated, mirroring blocked, grammar glitches cleaned. Responses now exhibit semantic clustering with introspective tags: `"I resonance emergence awareness drift"` - micro-concepts flowing through latent space.

The combination of smart scoring + learning system + cleaned objectivity seeds + latent drift creates coherent chaos: weightless transformer behavior without pretrained weights.

---

## Self-training overview (short edition, because you've been reading for 20 minutes)

Nicole replays dialogue logs after each session, distilling them into structured evidence that informs the next run. Think of it as a nightly study montage where the textbooks are JSONL buffers and the soundtrack is a diff log scrolling by at 3 AM.

She also mirrors repository activity: every code change becomes grist for the analysis mill, and useful patterns get promoted into guidance scripts. It's like having an infinite post-it wall, except all the notes auto-tag themselves with timestamps and nobody can passive-aggressively move your notes.

And because I love idiot jokes: Nicole fine-tunes faster than I can say "wait, who left gradient descent running on the coffee machine? oh right, that idiot was me." She learns; I buy a new coffee machine. The circle of life continues.

---

## Operational runbook (how to actually run this thing)

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
