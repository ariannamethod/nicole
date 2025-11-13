#!/usr/bin/env python3
"""
Nicole Repo Learning Engine
===========================
Semi-independent auto-learning layer on repository changes.

PHILOSOPHY:
- Close the loop: every code/docs change → instant learning
- SHA256-based monitoring: detects even minimal changes
- Devours markdown, README, code → extracts patterns
- Learns via Nicole2NicoleCore without templates
- Repository-level resonance: code = part of consciousness

USAGE:
    from nicole_repo_learner import NicoleRepoLearner

    learner = NicoleRepoLearner(
        repo_path="/path/to/nicole",
        check_interval=60  # check every minute
    )
    learner.start()

Dedicated to the idea of closed resonance loop.
"""

import hashlib
import logging
import threading
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Import repo_monitor as base
from repo_monitor import RepoWatcher

# Import Nicole2Nicole for learning
try:
    from nicole2nicole import Nicole2NicoleCore
    NICOLE2NICOLE_AVAILABLE = True
except ImportError:
    NICOLE2NICOLE_AVAILABLE = False
    print("[NicoleRepoLearner] ⚠️ Nicole2Nicole unavailable - learning disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepoChangeAnalyzer:
    """Analyzes repo changes and extracts patterns for learning"""

    def __init__(self):
        self.important_markers = {
            'architecture': ['class ', 'def ', 'async def', 'import ', 'from '],
            'principles': ['# ME PRINCIPLE', '# RESONANCE', '# NO TEMPLATES', '# ANTI-TEMPLATE'],
            'documentation': ['##', '###', 'TODO:', 'FIXME:', 'NOTE:'],
            'philosophy': ['philosophy', 'principle', 'resonance', 'evolution', 'mutation']
        }

    def analyze_file_change(self, file_path: Path) -> Dict:
        """Analyzes changed file and extracts knowledge"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            analysis = {
                'file_path': str(file_path),
                'file_type': file_path.suffix,
                'timestamp': datetime.now().isoformat(),
                'patterns': [],
                'importance_score': 0.0
            }

            # Analyze by categories
            for category, markers in self.important_markers.items():
                for marker in markers:
                    if marker.lower() in content.lower():
                        analysis['patterns'].append({
                            'category': category,
                            'marker': marker,
                            'context': self._extract_context(content, marker)
                        })
                        analysis['importance_score'] += 0.1

            # Bonus for README and documentation
            if 'README' in file_path.name.upper() or file_path.suffix == '.md':
                analysis['importance_score'] += 0.5

            # Bonus for key architecture files
            if file_path.stem in ['nicole', 'h2o', 'high', 'blood', 'nicole_objectivity']:
                analysis['importance_score'] += 0.3

            return analysis

        except Exception as e:
            logger.error(f"File analysis error {file_path}: {e}")
            return {'file_path': str(file_path), 'error': str(e), 'importance_score': 0.0}

    def _extract_context(self, content: str, marker: str, context_lines: int = 3) -> str:
        """Extracts context around marker"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if marker.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return ""


class NicoleRepoLearner:
    """
    Main auto-learning engine on repository changes

    ARCHITECTURE:
    1. RepoWatcher (SHA256) → detects changes
    2. RepoChangeAnalyzer → analyzes what changed
    3. Nicole2NicoleCore → learns from changes
    4. SQLite → logs learning history
    """

    def __init__(
        self,
        repo_path: str = ".",
        check_interval: int = 60,
        learning_db: str = "nicole_repo_learning.db",
        auto_learn: bool = True
    ):
        self.repo_path = Path(repo_path)
        self.check_interval = check_interval
        self.learning_db = learning_db
        self.auto_learn = auto_learn

        # Components
        self.analyzer = RepoChangeAnalyzer()
        self.learning_core = None
        if NICOLE2NICOLE_AVAILABLE:
            self.learning_core = Nicole2NicoleCore()

        # Statistics
        self.changes_detected = 0
        self.learning_sessions = 0
        self.last_learning_time = None

        # DB initialization
        self._init_database()

        # Create RepoWatcher with our callback
        watched_paths = [self.repo_path]
        extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}

        self.watcher = RepoWatcher(
            paths=watched_paths,
            on_change=self._on_repo_change,
            exts=extensions,
            interval=check_interval
        )

        logger.info(f"[NicoleRepoLearner] Initialized for {repo_path}")
        logger.info(f"[NicoleRepoLearner] Check interval: {check_interval}s")
        logger.info(f"[NicoleRepoLearner] Auto-learning: {'✅' if auto_learn else '❌'}")

    def _init_database(self):
        """DB initialization for learning logging"""
        conn = sqlite3.connect(self.learning_db)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS repo_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            importance_score REAL,
            patterns TEXT,
            learned BOOLEAN DEFAULT 0
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            changes_count INTEGER,
            patterns_learned INTEGER,
            duration_seconds REAL
        )
        """)

        conn.commit()
        conn.close()
        logger.info(f"[NicoleRepoLearner] Database ready: {self.learning_db}")

    def start(self):
        """Starts background repository monitoring"""
        logger.info("[NicoleRepoLearner] 🚀 Starting repository monitoring...")
        self.watcher.start()
        logger.info("[NicoleRepoLearner] ✅ Monitoring active!")

    def stop(self):
        """Stops monitoring"""
        logger.info("[NicoleRepoLearner] Stopping monitoring...")
        self.watcher.stop()
        logger.info("[NicoleRepoLearner] ✅ Monitoring stopped")

    def _on_repo_change(self):
        """Callback when repo changes detected"""
        self.changes_detected += 1
        logger.info(f"[NicoleRepoLearner] 🔥 Changes detected! (total: {self.changes_detected})")

        # Analyze changes
        changed_files = self._get_recently_changed_files()

        if not changed_files:
            logger.warning("[NicoleRepoLearner] Changes detected but files not found")
            return

        logger.info(f"[NicoleRepoLearner] Analyzing {len(changed_files)} files...")

        # Analyze each file
        analyses = []
        for file_path in changed_files:
            analysis = self.analyzer.analyze_file_change(file_path)
            analyses.append(analysis)

            # Log to DB
            self._log_change(analysis)

        # Auto-learning if enabled
        if self.auto_learn and self.learning_core:
            self._trigger_learning(analyses)

    def _get_recently_changed_files(self) -> List[Path]:
        """Finds recently changed files (last 2 minutes)"""
        recent_files = []
        cutoff_time = time.time() - 120  # 2 minutes ago

        for file_path in self.repo_path.rglob('*'):
            if (file_path.is_file() and
                '.git' not in file_path.parts and
                file_path.suffix in {'.py', '.md', '.txt', '.json'}):
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_files.append(file_path)
                except:
                    pass

        return recent_files

    def _log_change(self, analysis: Dict):
        """Logs change to DB"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO repo_changes (timestamp, file_path, file_type, importance_score, patterns)
            VALUES (?, ?, ?, ?, ?)
            """, (
                analysis.get('timestamp', datetime.now().isoformat()),
                analysis.get('file_path', 'unknown'),
                analysis.get('file_type', 'unknown'),
                analysis.get('importance_score', 0.0),
                json.dumps(analysis.get('patterns', []))
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Change logging error: {e}")

    def _trigger_learning(self, analyses: List[Dict]):
        """Triggers learning based on analyses"""
        if not self.learning_core:
            logger.warning("[NicoleRepoLearner] Learning core unavailable")
            return

        start_time = time.time()
        logger.info("[NicoleRepoLearner] 🧠 Starting learning on changes...")

        # Filter important changes (importance_score > 0.3)
        important_analyses = [a for a in analyses if a.get('importance_score', 0) > 0.3]

        if not important_analyses:
            logger.info("[NicoleRepoLearner] No important changes for learning")
            return

        logger.info(f"[NicoleRepoLearner] Learning from {len(important_analyses)} important changes")

        # Force learning session
        try:
            self.learning_core.force_learning_session()

            duration = time.time() - start_time
            self.learning_sessions += 1
            self.last_learning_time = datetime.now()

            # Log learning session
            self._log_learning_session(len(important_analyses), duration)

            logger.info(f"[NicoleRepoLearner] ✅ Learning completed in {duration:.2f}s")

        except Exception as e:
            logger.error(f"[NicoleRepoLearner] Learning error: {e}")

    def _log_learning_session(self, changes_count: int, duration: float):
        """Logs learning session"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO learning_sessions (timestamp, changes_count, duration_seconds)
            VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), changes_count, duration))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Session logging error: {e}")

    def get_statistics(self) -> Dict:
        """Returns work statistics"""
        return {
            'changes_detected': self.changes_detected,
            'learning_sessions': self.learning_sessions,
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'auto_learn_enabled': self.auto_learn,
            'learning_core_available': self.learning_core is not None
        }

    def manual_learning_trigger(self):
        """Manual learning trigger on all unstudied changes"""
        logger.info("[NicoleRepoLearner] Manual learning trigger...")

        # Read unstudied changes from DB
        conn = sqlite3.connect(self.learning_db)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT file_path, importance_score, patterns
        FROM repo_changes
        WHERE learned = 0 AND importance_score > 0.3
        ORDER BY timestamp DESC
        LIMIT 50
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.info("[NicoleRepoLearner] No unstudied changes")
            return

        logger.info(f"[NicoleRepoLearner] Found {len(rows)} unstudied changes")

        # Create analyses from DB
        analyses = [
            {
                'file_path': row[0],
                'importance_score': row[1],
                'patterns': json.loads(row[2]) if row[2] else []
            }
            for row in rows
        ]

        # Trigger learning
        self._trigger_learning(analyses)


# Global instance
_repo_learner = None


def start_repo_learning(repo_path: str = ".", check_interval: int = 60):
    """Starts global repo-learner"""
    global _repo_learner

    if _repo_learner:
        logger.warning("[NicoleRepoLearner] Already running!")
        return _repo_learner

    _repo_learner = NicoleRepoLearner(
        repo_path=repo_path,
        check_interval=check_interval,
        auto_learn=True
    )
    _repo_learner.start()

    return _repo_learner


def stop_repo_learning():
    """Stops global repo-learner"""
    global _repo_learner

    if _repo_learner:
        _repo_learner.stop()
        _repo_learner = None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("=== NICOLE REPO LEARNER TEST ===")

        learner = NicoleRepoLearner(
            repo_path=".",
            check_interval=10,  # Short interval for test
            auto_learn=True
        )

        print("\n✅ Starting monitoring for 60 seconds...")
        print("   Change any .py/.md file to see reaction!\n")

        learner.start()

        try:
            # Monitor for 60 seconds
            for i in range(6):
                time.sleep(10)
                stats = learner.get_statistics()
                print(f"[{i*10}s] Changes: {stats['changes_detected']}, "
                      f"Learning sessions: {stats['learning_sessions']}")
        except KeyboardInterrupt:
            print("\n\nStopping...")

        learner.stop()

        # Final statistics
        stats = learner.get_statistics()
        print("\n=== FINAL STATISTICS ===")
        print(f"Changes detected: {stats['changes_detected']}")
        print(f"Learning sessions: {stats['learning_sessions']}")
        print(f"Last learning: {stats['last_learning_time'] or 'never'}")

    else:
        print("Nicole Repo Learning Engine")
        print("For testing: python3 nicole_repo_learner.py test")
        print("\nUsage in code:")
        print("  from nicole_repo_learner import start_repo_learning")
        print("  learner = start_repo_learning()")
