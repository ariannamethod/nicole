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
1. [English-only boundary](#english-only-boundary-or-why-were-linguistic-fascists)
2. [Git signal](#git-signal-or-what-fresh-hell-happened-this-week)
3. [Core principles](#core-principles-the-philosophical-nonsense)
4. [Architecture panorama](#architecture-panorama-the-grand-tour-of-chaos)
5. [Conversation lifecycle](#conversation-lifecycle-six-acts-of-digital-theater)
6. [Compiler triad](#compiler-triad-three-compilers-enter-one-thought-emerges)
7. [Operational substrate (AMLK)](#operational-substrate-amlk-the-kernel-that-said-no-to-chaos)
8. [Module reference](#module-reference-the-parts-and-their-peculiarities)
9. [Language guardrails (deep dive)](#language-guardrails-deep-dive)
10. [Memory, metrics, and objectivity](#memory-metrics-and-objectivity)
11. [Repo-coupled evolution](#repo-coupled-evolution-she-learns-from-her-own-commits)
12. [Recent enhancements](#recent-enhancements-the-victory-lap)
13. [Self-training overview](#self-training-overview-short-edition)
14. [Operational runbook](#operational-runbook-how-to-actually-run-this-thing)
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

The repository is a delightful mix of orchestration, compilers, analytics, and operational tooling. Here's the labyrinth, organized by responsibility so you don't get lost:

### Cognitive spine (the thinky bits)

- `nicole.py` – Spins up transient transformer graphs, allocates ephemeral tensors like they're going out of style, and manages dialogue flow. The conductor of this orchestra.
  
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

