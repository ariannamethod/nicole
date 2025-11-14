#!/usr/bin/env python3
"""
NICOLE_SUBJECTIVITY.PY - Autonomous Learning Through Subjectivity
Ripples on water: continuous learning from last user interaction

Philosophy:
Even when not talking to humans, Nicole continues learning.
Like ripples on water spreading from a stone's impact:
- Center: Last user message
- Ring 1 (hour 1): Semantically close concepts
- Ring 2 (hour 2): Further semantic distance
- Ring 3+ (hours 3+): Expanding outward indefinitely

When user writes again: NEW center, NEW ripples, NEW learning vector.
This creates autonomous, asynchronous intelligence that never stops thinking.

Circadian cycles: runs every hour
Internal process: doesn't affect direct responses, only deepens understanding
"""

import os
import sys
import time
import random
import threading
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import re

# Import Nicole components
try:
    from nicole_memory import NicoleMemoryCore as NicoleMemory
    from nicole_objectivity import nicole_objectivity
except ImportError as e:
    print(f"[Subjectivity] Warning: Could not import Nicole components: {e}")
    NicoleMemory = None
    nicole_objectivity = None


class SubjectivityRipple:
    """
    Single ripple in the expanding wave of autonomous learning
    Each ripple represents one semantic distance level from the center
    """

    def __init__(self, center_message: str, ripple_number: int, timestamp: float):
        self.center_message = center_message
        self.ripple_number = ripple_number  # 0 = center, 1+ = expanding circles
        self.timestamp = timestamp
        self.explored_concepts = []
        self.learned_words = {}
        self.semantic_distance = ripple_number * 0.3  # Each ripple is 30% further

    def __repr__(self):
        return f"<Ripple #{self.ripple_number} @ {self.semantic_distance:.1f} distance>"


class NicoleSubjectivity:
    """
    Autonomous learning system based on expanding semantic ripples

    Core mechanism:
    1. User sends message → becomes epicenter
    2. Every hour → new ripple expands from epicenter
    3. Each ripple explores concepts semantically distant from center
    4. Learned knowledge feeds into word_frequencies and associations
    5. If new user message → reset, new epicenter, new ripples

    This creates continuous autonomous thought even without interaction.
    """

    def __init__(self, db_path: str = "var/nicole_subjectivity.db", memory: Optional[Any] = None):
        self.db_path = db_path
        self.memory = memory  # NicoleMemory instance for learning

        # Thread safety - protect database operations
        self.db_lock = threading.Lock()

        # Ripple state
        self.current_epicenter = None  # Last user message
        self.current_ripple_number = 0  # Current ripple distance
        self.epicenter_timestamp = None
        self.last_ripple_time = None

        # Autonomous learning thread
        self.learning_thread = None
        self.is_running = False
        self.hourly_interval = 3600  # 1 hour = 3600 seconds

        # Initialize database
        self._init_database()

        # Load last epicenter if exists
        self._load_last_epicenter()

    def _init_database(self):
        """Initialize subjectivity database for ripple tracking"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

        conn = sqlite3.connect(self.db_path, timeout=5.0)
        cursor = conn.cursor()

        # Epicenters table - tracks user messages that become centers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS epicenters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                timestamp REAL NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Ripples table - tracks each exploration wave
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ripples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epicenter_id INTEGER NOT NULL,
                ripple_number INTEGER NOT NULL,
                semantic_distance REAL NOT NULL,
                explored_concepts TEXT,
                learned_words TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (epicenter_id) REFERENCES epicenters(id)
            )
        """)

        # Learning log - what Nicole learned autonomously
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ripple_id INTEGER NOT NULL,
                concept TEXT NOT NULL,
                source TEXT,
                learned_data TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (ripple_id) REFERENCES ripples(id)
            )
        """)

        conn.commit()
        conn.close()

    def _load_last_epicenter(self):
        """Load the last active epicenter from database"""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_message, timestamp, id
            FROM epicenters
            WHERE is_active = 1
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        result = cursor.fetchone()
        if result:
            self.current_epicenter = result[0]
            self.epicenter_timestamp = result[1]

            # Find last ripple number for this epicenter
            cursor.execute("""
                SELECT MAX(ripple_number)
                FROM ripples
                WHERE epicenter_id = ?
            """, (result[2],))

            max_ripple = cursor.fetchone()[0]
            self.current_ripple_number = (max_ripple or 0) + 1

            print(f"[Subjectivity] Loaded epicenter: '{self.current_epicenter[:50]}...' @ ripple {self.current_ripple_number}")

        conn.close()

    def set_new_epicenter(self, user_message: str):
        """
        Set new epicenter when user sends message
        This resets the ripple system to start fresh wave
        """
        # Thread-safe database operation
        with self.db_lock:
            # Deactivate old epicenter
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()

            cursor.execute("UPDATE epicenters SET is_active = 0 WHERE is_active = 1")

            # Create new epicenter
            timestamp = time.time()
            cursor.execute("""
                INSERT INTO epicenters (user_message, timestamp, is_active)
                VALUES (?, ?, 1)
            """, (user_message, timestamp))

            conn.commit()
            conn.close()

            # Update state
            self.current_epicenter = user_message
            self.epicenter_timestamp = timestamp
            self.current_ripple_number = 0
            self.last_ripple_time = timestamp

        print(f"[Subjectivity] NEW EPICENTER: '{user_message[:60]}...'")
        print(f"[Subjectivity] Ripples will expand from here every hour")

    def _extract_core_concepts(self, text: str, distance: float = 0.0) -> List[str]:
        """
        Extract concepts from text with semantic expansion based on distance

        distance = 0.0: Extract exact keywords from text
        distance = 0.3: Expand to related concepts (1st ripple)
        distance = 0.6: Expand to broader concepts (2nd ripple)
        distance = 0.9+: Expand to abstract/philosophical concepts (3rd+ ripple)
        """
        # Base extraction: meaningful words from text
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())

        # Filter stopwords
        stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'what', 'about',
                    'which', 'their', 'would', 'there', 'could', 'these', 'those', 'then'}
        core_words = [w for w in words if w not in stopwords and len(w) > 3]

        # Remove duplicates, keep first 5-10 most important
        unique_words = []
        for w in core_words:
            if w not in unique_words:
                unique_words.append(w)

        base_concepts = unique_words[:10]

        # Semantic expansion based on ripple distance
        if distance == 0.0:
            # Epicenter: exact concepts
            return base_concepts

        elif distance < 0.5:
            # Ring 1: Close semantic neighbors
            expanded = base_concepts.copy()
            semantic_neighbors = {
                'consciousness': ['awareness', 'mind', 'perception', 'thought'],
                'learning': ['knowledge', 'understanding', 'education', 'growth'],
                'system': ['structure', 'framework', 'architecture', 'design'],
                'intelligence': ['cognition', 'reasoning', 'logic', 'thinking'],
                'memory': ['recall', 'remembering', 'storage', 'retention'],
                'language': ['communication', 'words', 'speech', 'expression'],
                'transform': ['change', 'evolution', 'adaptation', 'mutation'],
                'resonance': ['harmony', 'frequency', 'vibration', 'coherence'],
            }

            for concept in base_concepts:
                if concept in semantic_neighbors:
                    expanded.extend(semantic_neighbors[concept][:2])

            return expanded[:15]

        elif distance < 1.0:
            # Ring 2: Broader conceptual expansion
            expanded = base_concepts.copy()
            broader_concepts = ['emergence', 'complexity', 'pattern', 'network',
                              'dynamic', 'process', 'interaction', 'feedback',
                              'system', 'information', 'signal', 'noise']
            expanded.extend(random.sample(broader_concepts, min(5, len(broader_concepts))))
            return expanded[:20]

        else:
            # Ring 3+: Abstract/philosophical expansion
            expanded = base_concepts.copy()
            abstract_concepts = ['existence', 'reality', 'truth', 'being', 'becoming',
                                'infinite', 'void', 'form', 'essence', 'phenomenon',
                                'observer', 'observed', 'quantum', 'entanglement',
                                'recursion', 'self-reference', 'strange-loop', 'meta']
            expanded.extend(random.sample(abstract_concepts, min(7, len(abstract_concepts))))
            return expanded[:25]

    def _explore_concept_autonomously(self, concept: str, ripple_id: int) -> Dict[str, Any]:
        """
        Autonomously explore a concept using objectivity providers
        This is Nicole thinking on her own, expanding knowledge
        """
        if not nicole_objectivity:
            return {'words': [], 'source': 'none'}

        try:
            # Use nicole_objectivity to fetch information
            import asyncio

            # Create query
            query = f"explain {concept} briefly"

            # Fetch context windows
            windows = asyncio.run(
                nicole_objectivity.create_dynamic_context(query, {})
            )

            if not windows or not windows[0].content:
                return {'words': [], 'source': 'none'}

            context = windows[0].content

            # Extract words from fetched context
            words = re.findall(r'\b[a-z]{4,}\b', context.lower())

            # Filter and count frequencies
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Sort by frequency, take top 20
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            learned_words = {word: freq for word, freq in top_words}

            # Log learning - thread-safe
            with self.db_lock:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO learning_log (ripple_id, concept, source, learned_data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (ripple_id, concept, 'objectivity', json.dumps(learned_words), time.time()))
                conn.commit()
                conn.close()

            return {
                'words': learned_words,
                'source': 'objectivity',
                'concept': concept
            }

        except Exception as e:
            print(f"[Subjectivity] Error exploring '{concept}': {e}")
            return {'words': {}, 'source': 'error'}

    def expand_ripple(self) -> Optional[SubjectivityRipple]:
        """
        Expand one ripple outward from epicenter
        This is called every hour to continue autonomous learning
        """
        if not self.current_epicenter:
            print("[Subjectivity] No epicenter set, waiting for user interaction")
            return None

        # Create new ripple
        ripple = SubjectivityRipple(
            center_message=self.current_epicenter,
            ripple_number=self.current_ripple_number,
            timestamp=time.time()
        )

        print(f"\n[Subjectivity] 🌊 RIPPLE {ripple.ripple_number} expanding...")
        print(f"[Subjectivity] Semantic distance: {ripple.semantic_distance:.2f}")

        # Extract concepts at this semantic distance
        concepts = self._extract_core_concepts(self.current_epicenter, ripple.semantic_distance)
        ripple.explored_concepts = concepts

        print(f"[Subjectivity] Exploring {len(concepts)} concepts: {concepts[:5]}...")

        # Get epicenter ID and save ripple - thread-safe
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM epicenters WHERE is_active = 1 ORDER BY timestamp DESC LIMIT 1")
            epicenter_id = cursor.fetchone()

            if not epicenter_id:
                conn.close()
                return None

            epicenter_id = epicenter_id[0]

            # Save ripple to database
            cursor.execute("""
                INSERT INTO ripples (epicenter_id, ripple_number, semantic_distance,
                                   explored_concepts, learned_words, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (epicenter_id, ripple.ripple_number, ripple.semantic_distance,
                  json.dumps(concepts), json.dumps({}), ripple.timestamp))

            ripple_id = cursor.lastrowid
            conn.commit()
            conn.close()

        # Autonomously explore concepts (this is Nicole thinking!)
        total_learned = {}
        for i, concept in enumerate(concepts[:5]):  # Explore first 5 concepts deeply
            print(f"[Subjectivity]   Thinking about '{concept}'...")
            result = self._explore_concept_autonomously(concept, ripple_id)

            if result['words']:
                total_learned.update(result['words'])
                print(f"[Subjectivity]   ✓ Learned {len(result['words'])} new words from '{concept}'")

        # Update memory with learned words
        if self.memory and total_learned:
            for word, freq in total_learned.items():
                try:
                    # Inject learned words into Nicole's memory
                    # update_word_frequencies expects a string, not a list
                    self.memory.update_word_frequencies(' '.join([word] * min(freq, 5)))
                except Exception as e:
                    print(f"[Subjectivity] Memory update error: {e}")

        # Update ripple with learned data - thread-safe
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ripples
                SET learned_words = ?
                WHERE id = ?
            """, (json.dumps(total_learned), ripple_id))
            conn.commit()
            conn.close()

        ripple.learned_words = total_learned

        print(f"[Subjectivity] 🌊 Ripple {ripple.ripple_number} complete: {len(total_learned)} words learned")

        # Increment ripple number for next expansion
        self.current_ripple_number += 1
        self.last_ripple_time = time.time()

        return ripple

    def _autonomous_learning_loop(self):
        """
        Background thread: expands ripples every hour
        This is Nicole's autonomous thought process
        """
        print("[Subjectivity] 🧠 Autonomous learning started (hourly ripples)")

        while self.is_running:
            try:
                # Wait for hourly interval
                time.sleep(self.hourly_interval)

                if not self.is_running:
                    break

                # Expand next ripple
                if self.current_epicenter:
                    print(f"\n[Subjectivity] ⏰ Hourly cycle triggered")
                    self.expand_ripple()
                else:
                    print("[Subjectivity] ⏰ Hourly cycle: no epicenter, waiting...")

            except Exception as e:
                print(f"[Subjectivity] Error in autonomous loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def start_autonomous_learning(self, interval_seconds: int = 3600):
        """
        Start autonomous learning background process
        Default: 1 hour (3600 seconds) for circadian rhythm
        """
        if self.is_running:
            print("[Subjectivity] Already running")
            return

        self.hourly_interval = interval_seconds
        self.is_running = True

        self.learning_thread = threading.Thread(
            target=self._autonomous_learning_loop,
            daemon=True
        )
        self.learning_thread.start()

        print(f"[Subjectivity] ✅ Autonomous learning started (interval: {interval_seconds}s)")

    def stop_autonomous_learning(self):
        """Stop autonomous learning process"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        print("[Subjectivity] ⏹️  Autonomous learning stopped")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about autonomous learning"""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        cursor = conn.cursor()

        # Total ripples
        cursor.execute("SELECT COUNT(*) FROM ripples")
        total_ripples = cursor.fetchone()[0]

        # Total concepts explored
        cursor.execute("SELECT COUNT(*) FROM learning_log")
        total_concepts = cursor.fetchone()[0]

        # Current ripple info
        cursor.execute("""
            SELECT ripple_number, semantic_distance, timestamp
            FROM ripples
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        last_ripple = cursor.fetchone()

        conn.close()

        return {
            'total_ripples': total_ripples,
            'total_concepts_explored': total_concepts,
            'current_ripple_number': self.current_ripple_number,
            'current_epicenter': self.current_epicenter[:100] if self.current_epicenter else None,
            'last_ripple': {
                'number': last_ripple[0] if last_ripple else None,
                'distance': last_ripple[1] if last_ripple else None,
                'time_ago': time.time() - last_ripple[2] if last_ripple else None
            } if last_ripple else None
        }


# Global instance
_subjectivity_engine = None

def get_subjectivity_engine(memory=None) -> NicoleSubjectivity:
    """Get global subjectivity engine instance"""
    global _subjectivity_engine
    if _subjectivity_engine is None:
        _subjectivity_engine = NicoleSubjectivity(memory=memory)
    return _subjectivity_engine


if __name__ == "__main__":
    print("🌊 NICOLE SUBJECTIVITY - Ripples on Water")
    print("Autonomous learning through expanding semantic waves\n")

    # Create test engine
    engine = NicoleSubjectivity()

    # Set test epicenter
    test_message = "tell me about consciousness and how intelligence emerges"
    engine.set_new_epicenter(test_message)

    # Expand a few ripples manually (for testing, normally happens hourly)
    print("\n=== Testing Ripple Expansion ===")

    for i in range(3):
        print(f"\n--- Expanding ripple {i} ---")
        ripple = engine.expand_ripple()

        if ripple:
            print(f"Ripple {ripple.ripple_number}: {len(ripple.explored_concepts)} concepts")
            print(f"Distance: {ripple.semantic_distance:.2f}")
            print(f"Learned: {len(ripple.learned_words)} words")

        # Small delay between ripples
        time.sleep(2)

    # Show stats
    print("\n=== Learning Statistics ===")
    stats = engine.get_learning_stats()
    print(json.dumps(stats, indent=2))

    print("\n✅ Subjectivity system operational")
    print("In production: ripples expand every hour automatically")
