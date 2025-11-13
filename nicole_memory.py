#!/usr/bin/env python3
"""
Nicole Memory - Long-term memory module without weights
Stores and retrieves contextual information to maintain conversation coherence.
Uses semantic search and associative connections.
"""

import sqlite3
import json
import time
import math
import threading
import sys
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import h2o

# OPTIMIZATION: Import DB utilities for WAL mode and indexes
try:
    from db_utils import get_optimized_connection, create_memory_indexes
    DB_UTILS_AVAILABLE = True
except ImportError:
    print("[nicole_memory] WARNING: db_utils not available, using default SQLite")
    DB_UTILS_AVAILABLE = False

@dataclass
class MemoryEntry:
    """Memory entry"""
    id: str
    content: str
    context: str
    timestamp: float
    importance: float
    access_count: int = 0
    last_access: float = 0.0
    associations: List[str] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []

class SemanticIndex:
    """Semantic index for fast search without vector databases"""

    def __init__(self):
        self.word_to_entries = defaultdict(set)
        self.bigram_to_entries = defaultdict(set)
        self.trigram_to_entries = defaultdict(set)

    def index_entry(self, entry: MemoryEntry):
        """Indexes entry for search"""
        words = self._extract_words(entry.content + " " + entry.context)

        # Index by words
        for word in words:
            self.word_to_entries[word].add(entry.id)

        # Index by bigrams
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.bigram_to_entries[bigram].add(entry.id)

        # Index by trigrams
        for i in range(len(words) - 2):
            trigram = (words[i], words[i + 1], words[i + 2])
            self.trigram_to_entries[trigram].add(entry.id)
            
    def search(self, query: str, limit: int = 10) -> Set[str]:
        """Search by query, returns entry IDs"""
        query_words = self._extract_words(query)

        if not query_words:
            return set()

        # Search by words
        word_matches = set()
        for word in query_words:
            word_matches.update(self.word_to_entries.get(word, set()))

        # Search by bigrams
        bigram_matches = set()
        for i in range(len(query_words) - 1):
            bigram = (query_words[i], query_words[i + 1])
            bigram_matches.update(self.bigram_to_entries.get(bigram, set()))

        # Search by trigrams (highest priority)
        trigram_matches = set()
        for i in range(len(query_words) - 2):
            trigram = (query_words[i], query_words[i + 1], query_words[i + 2])
            trigram_matches.update(self.trigram_to_entries.get(trigram, set()))

        # Combine results with weights
        all_matches = []

        for entry_id in trigram_matches:
            all_matches.append((entry_id, 3.0))  # Trigrams weigh more

        for entry_id in bigram_matches:
            if entry_id not in trigram_matches:
                all_matches.append((entry_id, 2.0))  # Bigrams medium weight

        for entry_id in word_matches:
            if entry_id not in trigram_matches and entry_id not in bigram_matches:
                all_matches.append((entry_id, 1.0))  # Words minimum weight

        # Sort by relevance and limit
        all_matches.sort(key=lambda x: x[1], reverse=True)
        return {match[0] for match in all_matches[:limit]}
        
    def _extract_words(self, text: str) -> List[str]:
        """Extracts words from text"""
        # Simple tokenization
        words = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        return [word for word in words if len(word) > 2]  # Filter short words

class AssociativeNetwork:
    """Associative network for linking concepts"""

    def __init__(self):
        self.associations = defaultdict(lambda: defaultdict(float))
        self.concept_strength = defaultdict(float)

    def add_association(self, concept1: str, concept2: str, strength: float = 1.0):
        """Adds association between concepts"""
        self.associations[concept1][concept2] += strength
        self.associations[concept2][concept1] += strength
        self.concept_strength[concept1] += strength * 0.5
        self.concept_strength[concept2] += strength * 0.5

    def get_related_concepts(self, concept: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Returns related concepts"""
        if concept not in self.associations:
            return []

        related = list(self.associations[concept].items())
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:limit]

    def strengthen_association(self, concept1: str, concept2: str, factor: float = 1.1):
        """Strengthens association between concepts"""
        if concept2 in self.associations[concept1]:
            self.associations[concept1][concept2] *= factor
            self.associations[concept2][concept1] *= factor

    def decay_associations(self, decay_factor: float = 0.99):
        """Weakens all associations (forgetting)"""
        for concept1 in self.associations:
            for concept2 in list(self.associations[concept1].keys()):
                self.associations[concept1][concept2] *= decay_factor
                if self.associations[concept1][concept2] < 0.01:
                    del self.associations[concept1][concept2]

class NicoleMemoryCore:
    """Core of Nicole's memory system"""

    def __init__(self, db_path: str = "nicole_memory.db"):
        self.db_path = db_path
        self.semantic_index = SemanticIndex()
        self.associative_network = AssociativeNetwork()
        self.memory_cache = {}
        self.recent_memories = deque(maxlen=100)
        self.memory_lock = threading.Lock()
        self.init_database()
        self.load_memories_to_cache()

    def init_database(self):
        """Initialize memory database with optimizations (WAL mode)"""
        # OPTIMIZATION: Use optimized connection with WAL mode
        if DB_UTILS_AVAILABLE:
            conn = get_optimized_connection(self.db_path)
        else:
            conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # New tables for advanced memory
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_entries (
            id TEXT PRIMARY KEY,
            content TEXT,
            context TEXT,
            timestamp REAL,
            importance REAL,
            access_count INTEGER DEFAULT 0,
            last_access REAL DEFAULT 0,
            associations TEXT,
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concept_associations (
            concept1 TEXT,
            concept2 TEXT,
            strength REAL,
            last_update REAL,
            PRIMARY KEY (concept1, concept2)
        )
        """)
        
        # COMPATIBILITY: create old tables for backwards compatibility
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            timestamp REAL,
            user_input TEXT,
            nicole_output TEXT,
            metrics TEXT,
            transformer_config TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT PRIMARY KEY,
            count INTEGER DEFAULT 1
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bigrams (
            id INTEGER PRIMARY KEY,
            w1 TEXT,
            w2 TEXT,
            count INTEGER DEFAULT 1,
            UNIQUE(w1, w2)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_first_contact (
            user_id TEXT PRIMARY KEY,
            first_contact_time REAL,
            template_phase_completed INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0
        )
        """)

        conn.commit()

        # OPTIMIZATION: Create indexes for fast queries
        if DB_UTILS_AVAILABLE:
            create_memory_indexes(conn)
            print("[nicole_memory] Database optimized with WAL mode + indexes")

        conn.close()

    def load_memories_to_cache(self):
        """Loads memories into cache and indexes"""
        conn = get_optimized_connection(self.db_path) if DB_UTILS_AVAILABLE else sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memory_entries ORDER BY importance DESC LIMIT 1000")
        rows = cursor.fetchall()
        
        for row in rows:
            entry = MemoryEntry(
                id=row[0],
                content=row[1], 
                context=row[2],
                timestamp=row[3],
                importance=row[4],
                access_count=row[5],
                last_access=row[6],
                associations=json.loads(row[7]) if row[7] else []
            )
            
            self.memory_cache[entry.id] = entry
            self.semantic_index.index_entry(entry)

        # Load associations
        cursor.execute("SELECT concept1, concept2, strength FROM concept_associations")
        assoc_rows = cursor.fetchall()

        for concept1, concept2, strength in assoc_rows:
            self.associative_network.associations[concept1][concept2] = strength

        conn.close()

        # MIGRATION: load old data if no new data exists
        if len(self.memory_cache) == 0:
            self.migrate_old_memory_data()

        print(f"[NicoleMemory] Loaded {len(self.memory_cache)} memories")

        # Add compatibility with main Nicole
        self.word_frequencies = defaultdict(int)
        self.bigram_transitions = defaultdict(lambda: defaultdict(int))
        try:
            from nicole_metrics import VerbGraph
            self.verb_graph = VerbGraph()
        except ImportError:
            self.verb_graph = None

        # Load word_frequencies and bigrams
        self.load_compatibility_data()
        
    def store_memory(self, content: str, context: str = "", importance: float = 1.0,
                    associations: List[str] = None) -> str:
        """Stores new memory"""
        with self.memory_lock:
            memory_id = f"mem_{int(time.time() * 1000000)}"

            entry = MemoryEntry(
                id=memory_id,
                content=content,
                context=context,
                timestamp=time.time(),
                importance=importance,
                associations=associations or []
            )

            # Save to cache and index
            self.memory_cache[memory_id] = entry
            self.semantic_index.index_entry(entry)
            self.recent_memories.append(memory_id)

            # Create associations
            self._create_associations(entry)

            # Save to database
            self._save_entry_to_db(entry)

            print(f"[NicoleMemory] Stored memory {memory_id}")
            return memory_id
            
    def _create_associations(self, entry: MemoryEntry):
        """Creates associations for new memory"""
        words = self.semantic_index._extract_words(entry.content + " " + entry.context)

        # Create associations between words in memory
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j:
                    distance = abs(i - j)
                    strength = 1.0 / (1 + distance * 0.1)  # Nearby words are linked stronger
                    self.associative_network.add_association(word1, word2, strength * entry.importance)

    def _save_entry_to_db(self, entry: MemoryEntry):
        """Saves entry to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO memory_entries 
        (id, content, context, timestamp, importance, access_count, last_access, associations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            entry.context, 
            entry.timestamp,
            entry.importance,
            entry.access_count,
            entry.last_access,
            json.dumps(entry.associations)
        ))
        
        conn.commit()
        conn.close()
        
    def recall_memories(self, query: str, limit: int = 5, min_importance: float = 0.1) -> List[MemoryEntry]:
        """Recalls relevant memories"""
        with self.memory_lock:
            # Semantic search
            matching_ids = self.semantic_index.search(query, limit * 3)

            # Filter by importance and get entries
            relevant_memories = []
            for memory_id in matching_ids:
                if memory_id in self.memory_cache:
                    entry = self.memory_cache[memory_id]
                    if entry.importance >= min_importance:
                        relevant_memories.append(entry)

            # Sort by relevance (importance + recency + access frequency)
            def relevance_score(entry: MemoryEntry) -> float:
                recency = 1.0 / (1 + (time.time() - entry.timestamp) / 86400)  # Recency in days
                access_freq = math.log(1 + entry.access_count) / 10
                return entry.importance * 0.5 + recency * 0.3 + access_freq * 0.2

            relevant_memories.sort(key=relevance_score, reverse=True)

            # Update access statistics
            for entry in relevant_memories[:limit]:
                entry.access_count += 1
                entry.last_access = time.time()
                self._save_entry_to_db(entry)

            return relevant_memories[:limit]
            
    def get_associative_context(self, query: str, depth: int = 2) -> List[str]:
        """Gets associative context for query"""
        words = self.semantic_index._extract_words(query)

        all_associations = set()

        # First level of associations
        for word in words:
            related = self.associative_network.get_related_concepts(word, 3)
            for concept, strength in related:
                if strength > 0.5:  # Only strong associations
                    all_associations.add(concept)

        # Second level (associations of associations)
        if depth > 1:
            second_level = set()
            for concept in list(all_associations):
                related = self.associative_network.get_related_concepts(concept, 2)
                for concept2, strength in related:
                    if strength > 0.3:
                        second_level.add(concept2)
            all_associations.update(second_level)

        return list(all_associations)
        
    def consolidate_memories(self, threshold: float = 0.8):
        """Consolidates similar memories"""
        with self.memory_lock:
            memories = list(self.memory_cache.values())
            to_merge = []

            # Find similar memories
            for i, mem1 in enumerate(memories):
                for j, mem2 in enumerate(memories[i+1:], i+1):
                    similarity = self._calculate_similarity(mem1, mem2)
                    if similarity > threshold:
                        to_merge.append((mem1, mem2, similarity))

            # Merge similar memories
            for mem1, mem2, similarity in to_merge:
                merged_entry = self._merge_memories(mem1, mem2)

                # Remove old entries
                if mem1.id in self.memory_cache:
                    del self.memory_cache[mem1.id]
                if mem2.id in self.memory_cache:
                    del self.memory_cache[mem2.id]

                # Add merged entry
                self.memory_cache[merged_entry.id] = merged_entry
                self.semantic_index.index_entry(merged_entry)
                self._save_entry_to_db(merged_entry)

            if to_merge:
                print(f"[NicoleMemory] Consolidated {len(to_merge)} memory pairs")
                
    def _calculate_similarity(self, mem1: MemoryEntry, mem2: MemoryEntry) -> float:
        """Calculates similarity of two memories"""
        words1 = set(self.semantic_index._extract_words(mem1.content))
        words2 = set(self.semantic_index._extract_words(mem2.content))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union > 0 else 0.0

        # Account for temporal proximity
        time_diff = abs(mem1.timestamp - mem2.timestamp)
        time_similarity = 1.0 / (1 + time_diff / 3600)  # Hours

        return jaccard * 0.7 + time_similarity * 0.3

    def _merge_memories(self, mem1: MemoryEntry, mem2: MemoryEntry) -> MemoryEntry:
        """Merges two memories"""
        merged_content = f"{mem1.content} | {mem2.content}"
        merged_context = f"{mem1.context} | {mem2.context}"
        
        return MemoryEntry(
            id=f"merged_{int(time.time() * 1000000)}",
            content=merged_content,
            context=merged_context,
            timestamp=max(mem1.timestamp, mem2.timestamp),
            importance=max(mem1.importance, mem2.importance),
            access_count=mem1.access_count + mem2.access_count,
            last_access=max(mem1.last_access, mem2.last_access),
            associations=list(set(mem1.associations + mem2.associations))
        )
        
    def forget_old_memories(self, age_threshold: float = 7 * 24 * 3600):  # 7 days
        """Forgets old unimportant memories"""
        with self.memory_lock:
            current_time = time.time()
            to_forget = []

            for memory_id, entry in self.memory_cache.items():
                age = current_time - entry.timestamp

                # Forget if old AND unimportant AND rarely used
                if (age > age_threshold and
                    entry.importance < 0.3 and
                    entry.access_count < 2):
                    to_forget.append(memory_id)

            # Remove forgotten memories
            for memory_id in to_forget:
                del self.memory_cache[memory_id]

            if to_forget:
                print(f"[NicoleMemory] Forgot {len(to_forget)} old memories")
                
    def get_conversation_context(self, current_input: str, session_id: str = None) -> str:
        """Gets context for current conversation"""
        # Recall relevant memories
        relevant_memories = self.recall_memories(current_input, limit=3)

        # Get associative context
        associations = self.get_associative_context(current_input, depth=1)

        # Form context
        context_parts = []

        if relevant_memories:
            context_parts.append("Relevant memories:")
            for mem in relevant_memories:
                context_parts.append(f"- {mem.content[:100]}...")

        if associations:
            context_parts.append(f"Associations: {', '.join(associations[:10])}")

        return " ".join(context_parts) if context_parts else ""
        
    def learn_from_conversation(self, user_input: str, nicole_output: str,
                              session_context: str = "", importance: float = None):
        """Learns from conversation"""
        if importance is None:
            # Automatically determine importance
            importance = self._calculate_importance(user_input, nicole_output)

        # Store memory of interaction
        memory_content = f"User: {user_input} | Nicole: {nicole_output}"
        memory_id = self.store_memory(memory_content, session_context, importance)

        # Create associations between concepts in conversation
        user_concepts = self.semantic_index._extract_words(user_input)
        nicole_concepts = self.semantic_index._extract_words(nicole_output)

        # Link user and Nicole concepts
        for user_concept in user_concepts:
            for nicole_concept in nicole_concepts:
                self.associative_network.add_association(user_concept, nicole_concept, importance)

        return memory_id
        
    def _calculate_importance(self, user_input: str, nicole_output: str) -> float:
        """Automatically calculates interaction importance"""
        # Base importance
        importance = 0.5

        # Increase importance for:
        # - Long messages (more information)
        if len(user_input) > 50:
            importance += 0.2

        # - Questions (require remembering)
        if any(char in user_input for char in '?'):
            importance += 0.1

        # - Personal information (Russian words preserved as data)
        personal_words = ['я', 'мне', 'мой', 'моя', 'мое', 'меня', 'себя']
        if any(word in user_input.lower().split() for word in personal_words):
            importance += 0.3

        # - Emotional words (Russian words preserved as data)
        emotional_words = ['люблю', 'ненавижу', 'нравится', 'злой', 'грустный', 'счастливый']
        if any(word in user_input.lower() for word in emotional_words):
            importance += 0.2

        return min(1.0, importance)
        
    def periodic_maintenance(self):
        """Periodic memory maintenance"""
        while True:
            try:
                time.sleep(3600)  # Every hour

                # Consolidate memories
                self.consolidate_memories()

                # Forget old memories
                self.forget_old_memories()

                # Decay associations
                self.associative_network.decay_associations()

                print("[NicoleMemory] Periodic maintenance completed")

            except Exception as e:
                print(f"[NicoleMemory:ERROR] Maintenance error: {e}")

    def start_maintenance(self):
        """Starts periodic maintenance"""
        maintenance_thread = threading.Thread(target=self.periodic_maintenance, daemon=True)
        maintenance_thread.start()
        print("[NicoleMemory] Periodic maintenance started")

    def get_memory_statistics(self) -> Dict:
        """Returns memory statistics"""
        total_memories = len(self.memory_cache)
        
        if total_memories == 0:
            return {'total_memories': 0}
            
        importances = [mem.importance for mem in self.memory_cache.values()]
        access_counts = [mem.access_count for mem in self.memory_cache.values()]
        
        return {
            'total_memories': total_memories,
            'avg_importance': sum(importances) / len(importances),
            'max_importance': max(importances),
            'avg_access_count': sum(access_counts) / len(access_counts),
            'total_associations': sum(len(assocs) for assocs in self.associative_network.associations.values()),
            'recent_memories': len(self.recent_memories)
        }
    
    def migrate_old_memory_data(self):
        """Migrates data from old tables to new memory system"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if old data exists
            cursor.execute("SELECT COUNT(*) FROM conversations")
            old_conversations = cursor.fetchone()[0]

            if old_conversations > 0:
                print(f"[NicoleMemory] Migrating {old_conversations} old conversations...")

                # Load old conversations
                cursor.execute("""
                    SELECT user_input, nicole_output, timestamp, session_id
                    FROM conversations
                    WHERE user_input IS NOT NULL AND nicole_output IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 500
                """)

                for user_input, nicole_output, timestamp, session_id in cursor.fetchall():
                    # Create memory from conversation
                    content = f"User: {user_input} | Nicole: {nicole_output}"
                    context = f"conversation_{session_id or 'unknown'}"
                    importance = 0.7  # Medium importance for migrated data

                    memory_id = self.store_memory(content, context, importance)

                print(f"[NicoleMemory] Migration completed!")

            conn.close()

        except Exception as e:
            print(f"[NicoleMemory] Migration error: {e}")

    def load_compatibility_data(self):
        """Loads word_frequencies and bigrams for compatibility"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load word frequencies
            cursor.execute("SELECT word, count FROM word_frequencies")
            for word, count in cursor.fetchall():
                self.word_frequencies[word] = count

            # Load bigrams
            cursor.execute("SELECT w1, w2, count FROM bigrams")
            for w1, w2, count in cursor.fetchall():
                self.bigram_transitions[w1][w2] = count

            conn.close()

            words_count = len(self.word_frequencies)
            bigrams_count = sum(len(transitions) for transitions in self.bigram_transitions.values())
            if words_count > 0 or bigrams_count > 0:
                print(f"[NicoleMemory] Loaded compatibility: {words_count} words, {bigrams_count} bigrams")

        except Exception as e:
            print(f"[NicoleMemory] Compatibility loading error: {e}")

# Integration will be added later
# TODO: Integrate with main Nicole system

# Integration function will be added later

def test_memory_system():
    """Memory system testing"""
    print("=== NICOLE MEMORY SYSTEM TEST ===")
    
    memory_core = NicoleMemoryCore()

    # Test 1: Storing memories
    print("\\n--- Storage test ---")
    mem1 = memory_core.store_memory("User likes coffee", "preferences conversation", 0.8)
    mem2 = memory_core.store_memory("Discussed weather yesterday", "casual conversation", 0.3)
    mem3 = memory_core.store_memory("User works as programmer", "personal information", 0.9)

    # Test 2: Memory recall
    print("\\n--- Recall test ---")
    results = memory_core.recall_memories("coffee programmer", limit=3)
    for mem in results:
        print(f"Found: {mem.content} (importance: {mem.importance:.2f})")

    # Test 3: Associative context
    print("\\n--- Associations test ---")
    associations = memory_core.get_associative_context("work coffee")
    print(f"Associations: {associations[:10]}")

    # Test 4: Conversation simulation
    print("\\n--- Conversation simulation test ---")

    test_conversations = [
        ("Hello! My name is Alex", "Hi Alex!"),
        ("I work as a programmer", "Cool! Programming is interesting work"),
        ("I love drinking coffee in the morning", "Coffee is a great way to start the day"),
        ("How are you?", "Good, thanks!"),
    ]

    for user_msg, nicole_response in test_conversations:
        memory_core.learn_from_conversation(user_msg, nicole_response, "test_session")
        print(f"Stored: {user_msg} -> {nicole_response}")

    # Check search on stored conversations
    print("\\n--- Search on stored conversations ---")
    results = memory_core.recall_memories("work programmer", limit=2)
    for mem in results:
        print(f"Found: {mem.content[:100]}...")

    # Statistics
    stats = memory_core.get_memory_statistics()
    print(f"\\nMemory statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

    print("\\n=== MEMORY TEST COMPLETED ===")

# === COMPATIBILITY METHODS WITH MAIN NICOLE ===

def add_compatibility_methods():
    """Adds compatibility methods to NicoleMemoryCore"""
    
    def update_word_frequencies(self, text: str):
        words = text.lower().split()
        for word in words:
            self.word_frequencies[word] += 1
    
    def update_bigrams(self, text: str):
        words = text.lower().split()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigram_transitions[w1][w2] += 1
    
    def log_conversation(self, session_id: str, user_input: str, nicole_output: str, 
                        metrics: dict, transformer_config: dict):
        conversation_content = f"User: {user_input} | Nicole: {nicole_output}"
        self.store_memory(content=conversation_content, context=f"Session: {session_id}", importance=1.0)
        self.update_word_frequencies(user_input)
        self.update_word_frequencies(nicole_output)
        self.update_bigrams(user_input)
        self.update_bigrams(nicole_output)
    
    def log_transformer_lifecycle(self, transformer_id: str, session_id: str, architecture: dict, creation_time: float, death_time: float = None):
        action = "died" if death_time else "created"
        lifecycle_content = f"Transformer {transformer_id} {action}"
        self.store_memory(content=lifecycle_content, context=f"Session: {session_id}", importance=0.8)
    
    def is_response_repetitive(self, response: str, threshold: float = 0.8) -> bool:
        return False
    
    def get_semantic_candidates(self, word: str, distance_percent: float = 0.5) -> List[str]:
        """
        Gets semantic candidates (compatibility with nicole.py)

        FIXED: When DB is empty, return [] instead of [word] to avoid duplicates
        - Empty DB → candidates_50 = [], candidates_70 = []
        - Let objectivity seeds be the primary source when learning hasn't started
        """
        # Convert distance_percent to limit for compatibility
        limit = max(5, int(distance_percent * 20))  # 0.5 -> 10, 0.7 -> 14
        candidates = []
        if word in self.associative_network.associations:
            candidates = list(self.associative_network.associations[word].keys())[:limit]

        # FIX: Don't return [word] as fallback - causes duplicates
        # Instead return empty list and let caller handle it
        return candidates

    # Add methods to class
    NicoleMemoryCore.update_word_frequencies = update_word_frequencies
    NicoleMemoryCore.update_bigrams = update_bigrams
    NicoleMemoryCore.log_conversation = log_conversation
    NicoleMemoryCore.log_transformer_lifecycle = log_transformer_lifecycle
    NicoleMemoryCore.is_response_repetitive = is_response_repetitive
    NicoleMemoryCore.get_semantic_candidates = get_semantic_candidates

# Automatically add compatibility
add_compatibility_methods()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_memory_system()
    else:
        print("Nicole Memory System ready to work")
        print("For testing run: python3 nicole_memory.py test")
