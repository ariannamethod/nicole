# Nicole System State Audit

## Overview
- Baseline synchronized with `claude/review-last-commit-01MVdkKFozkk2xfkg8LsBkvK` to restore the stable Nicole v0.3 behaviour.
- AMLK integration rolled back to the interactive `letsgo.py` gateway used in the working branch, removing the experimental direct API shim.
- Documentation refreshed to capture current module responsibilities, diagnostics, and operational guardrails.

## Core Modules Snapshot
### Generation Core (`high.py`)
- Latent Drift v0.4 cluster-based generation remains active, using resonance scoring and introspective tags for coherence.
- Maintains grammar enforcement and anti-mirroring safeguards during response construction.

### Subjectivity Module (`nicole_subjectivity.py`)
- Provides autonomous learning cycles with thread-safe SQLite access and background worker orchestration.
- Uses timed locks and frequency updates to avoid race conditions while mutating long-term resonance memory.

### Objectivity Module (`nicole_objectivity.py`)
- Supplies non-mirroring external context via Reddit/Wikipedia seeds with adaptive filtering of user echoes.
- Supports semantic retrieval with strict noise suppression to stabilise grounding data.

### Memory Systems (`nicole_memory.py`)
- Centralised `NicoleMemoryCore` handles shared state while preventing duplicate activation of the High pipeline.
- Provides safe fallbacks for memory word updates and gracefully logs activation issues.

### Repo Learning (`nicole_repo_learner.py` & `repo_monitor.py`)
- File watcher feeds Markdown diffs into learning routines so Nicole can ingest repository evolution autonomously.
- Combines snapshotting, adaptive weighting, and throttled learning windows to prevent feedback loops.

## AMLK Integration Status (`nicole_amlk.py`)
- Operates through the legacy subprocess bridge that boots AMLK's `letsgo.py` console when required.
- Communicates with AMLK via stdin/stdout queues guarded by threading locks, enabling synchronous command dispatch.
- Provides convenience wrappers for file, process, and network operations while logging outcomes for diagnostics.

## Diagnostics & Safety
- Extensive logging pipeline ensures subjectivity/objectivity/High subsystems report failures without impacting user responses.
- Graceful shutdown routines for AMLK prevent orphaned processes and keep the system ready for repeated activations.
- Repo monitor and memory modules include guardrails against concurrent writes to maintain database integrity.

## Testing
- Run `pytest -q` to execute the automated regression suite (`test_quick_wins.py`).
- Ensure the environment exposes necessary API keys when exercising external integrations (Reddit/Wikipedia) to avoid network-related false negatives.

