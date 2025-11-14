#!/usr/bin/env python3
"""
Nicole RAG - Retrieval Augmented Generation without vector databases
Search and context augmentation system for Nicole.
"""

import sqlite3
import json
import time
import math
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import h2o

class ChaoticRetriever:
    """Chaotic context retriever"""

    def __init__(self, memory_db: str = "nicole_memory.db"):
        self.memory_db = memory_db
        self.chaos_factor = 0.1  # Basic chaos element in search
        self.retrieval_patterns = {}
        self.user_chaos_preferences = {}  # ADAPTIVE: personal chaos per user
        
    def retrieve_context(self, query: str, chaos_level: float = None, limit: int = 8) -> List[Dict]:
        """Retrieves context with chaos element"""
        if chaos_level is None:
            chaos_level = self.chaos_factor
            
        # Normal search
        normal_results = self._semantic_search(query, limit=10)
        
        # Chaotic search (random connections)
        chaotic_results = self._chaotic_search(query, chaos_level)
        
        # Combine results
        all_results = normal_results + chaotic_results
        
        # Remove duplicates and sort
        unique_results = {}
        for result in all_results:
            if result['id'] not in unique_results:
                unique_results[result['id']] = result
            else:
                # Increase relevance for duplicates
                unique_results[result['id']]['relevance'] += result['relevance'] * 0.5
                
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return final_results[:limit]
        
    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Normal semantic search"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Search in conversations
        cursor.execute("""
        SELECT user_input, nicole_output, timestamp, metrics 
        FROM conversations 
        WHERE user_input LIKE ? OR nicole_output LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            user_input, nicole_output, timestamp, metrics = row

            # TEMPORAL: pass timestamp for temporal weighting
            relevance = self._calculate_relevance(query, user_input + " " + nicole_output, timestamp=timestamp)

            results.append({
                'id': f"conv_{timestamp}",
                'content': f"User: {user_input} | Nicole: {nicole_output}",
                'type': 'conversation',
                'relevance': relevance,
                'timestamp': timestamp
            })
            
        conn.close()
        return results
        
    def _chaotic_search(self, query: str, chaos_level: float) -> List[Dict]:
        """Chaotic search with unexpected connections"""
        if chaos_level <= 0:
            return []
            
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Random record search
        cursor.execute("""
        SELECT user_input, nicole_output, timestamp 
        FROM conversations 
        ORDER BY RANDOM() 
        LIMIT ?
        """, (int(20 * chaos_level),))
        
        chaotic_results = []
        for row in cursor.fetchall():
            user_input, nicole_output, timestamp = row

            # Random relevance with chaos element + temporal
            base_relevance = self._calculate_relevance(query, user_input + " " + nicole_output, timestamp=timestamp)
            chaos_boost = chaos_level * (0.5 + 0.5 * hash(user_input) % 100 / 100)
            
            chaotic_results.append({
                'id': f"chaos_{timestamp}",
                'content': f"[CHAOS] User: {user_input} | Nicole: {nicole_output}",
                'type': 'chaotic_memory',
                'relevance': base_relevance + chaos_boost,
                'timestamp': timestamp
            })
            
        conn.close()
        return chaotic_results
        
    def _calculate_relevance(self, query: str, content: str, timestamp: float = None) -> float:
        """
        Calculates content relevance to query + temporal weighting

        Added temporal weight: fresh memories more important than old
        Decay: e^(-age_days / 30) - half-life 30 days
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        intersection = len(query_words & content_words)
        union = len(query_words | content_words)

        jaccard = intersection / union if union > 0 else 0.0

        # Bonus for exact phrase matches
        exact_matches = sum(1 for word in query_words if word in content.lower())
        exact_bonus = exact_matches / len(query_words)

        base_relevance = jaccard * 0.7 + exact_bonus * 0.3

        # TEMPORAL WEIGHTING: fresh memories more important
        if timestamp:
            age_seconds = time.time() - timestamp
            age_days = age_seconds / 86400  # Seconds to days

            # Exponential decay: e^(-age / 30)
            # age=0 days: weight=1.0
            # age=30 days: weight=0.37
            # age=60 days: weight=0.14
            temporal_weight = math.exp(-age_days / 30)

            # Combine: 70% content relevance + 30% temporal weight
            final_relevance = base_relevance * 0.7 + temporal_weight * 0.3

            return final_relevance
        else:
            return base_relevance

    def adapt_chaos_from_feedback(self, user_id: str, feedback_score: float):
        """
        ADAPTIVE CHAOS: Adapts chaos_factor to user based on feedback

        Logic:
        - feedback > 0.7: user likes surprises → increase chaos
        - feedback < 0.3: user likes precision → decrease chaos

        Args:
            user_id: User ID
            feedback_score: success score [0.0 - 1.0]
        """
        if user_id not in self.user_chaos_preferences:
            self.user_chaos_preferences[user_id] = self.chaos_factor

        current_chaos = self.user_chaos_preferences[user_id]

        if feedback_score > 0.7:
            # User satisfied → increase chaos (more creativity)
            new_chaos = min(0.3, current_chaos * 1.1)
            print(f"[RAG:AdaptiveChaos] User {user_id}: feedback={feedback_score:.2f} → chaos {current_chaos:.2f} → {new_chaos:.2f} ↑")
        elif feedback_score < 0.3:
            # User dissatisfied → decrease chaos (more precision)
            new_chaos = max(0.05, current_chaos * 0.9)
            print(f"[RAG:AdaptiveChaos] User {user_id}: feedback={feedback_score:.2f} → chaos {current_chaos:.2f} → {new_chaos:.2f} ↓")
        else:
            # Normal → leave as is
            new_chaos = current_chaos

        self.user_chaos_preferences[user_id] = new_chaos

    def get_user_chaos_level(self, user_id: str) -> float:
        """Get user's personal chaos_factor"""
        return self.user_chaos_preferences.get(user_id, self.chaos_factor)

class ContextAugmenter:
    """Context augmenter for generation"""
    
    def __init__(self, retriever: ChaoticRetriever):
        self.retriever = retriever
        self.augmentation_strategies = {
            'factual': self._factual_augmentation,
            'creative': self._creative_augmentation,
            'chaotic': self._chaotic_augmentation,
            'balanced': self._balanced_augmentation
        }
        
    def augment_context(self, user_input: str, current_context: str = "", 
                       strategy: str = 'balanced') -> str:
        """Augments context for response generation"""
        
        # Get relevant context
        retrieved_context = self.retriever.retrieve_context(user_input)
        
        # Apply augmentation strategy
        augmentation_func = self.augmentation_strategies.get(strategy, self._balanced_augmentation)
        augmented_context = augmentation_func(user_input, current_context, retrieved_context)
        
        return augmented_context
        
    def _factual_augmentation(self, user_input: str, current_context: str, 
                            retrieved: List[Dict]) -> str:
        """Factual augmentation - only relevant facts"""
        factual_parts = [current_context] if current_context else []
        
        # Take only most relevant records
        for item in retrieved[:3]:
            if item['relevance'] > 0.5 and item['type'] == 'conversation':
                factual_parts.append(f"Context: {item['content']}")
                
        return " | ".join(factual_parts)
        
    def _creative_augmentation(self, user_input: str, current_context: str,
                             retrieved: List[Dict]) -> str:
        """Creative augmentation - unexpected connections"""
        creative_parts = [current_context] if current_context else []
        
        # Take diverse records for creativity
        for item in retrieved[::2]:  # Every second record
            creative_parts.append(f"Inspiration: {item['content']}")
            
        return " | ".join(creative_parts)
        
    def _chaotic_augmentation(self, user_input: str, current_context: str,
                            retrieved: List[Dict]) -> str:
        """Chaotic augmentation - full random"""
        chaotic_parts = [current_context] if current_context else []
        
        # Add random elements
        for item in retrieved:
            if item['type'] == 'chaotic_memory':
                chaotic_parts.append(f"Chaos: {item['content']}")
                
        return " | ".join(chaotic_parts)
        
    def _balanced_augmentation(self, user_input: str, current_context: str,
                             retrieved: List[Dict]) -> str:
        """Balanced augmentation"""
        balanced_parts = [current_context] if current_context else []
        
        # Facts (high relevance)
        facts = [item for item in retrieved if item['relevance'] > 0.6][:2]
        for fact in facts:
            balanced_parts.append(f"Fact: {fact['content']}")
            
        # Creative (medium relevance)
        creative = [item for item in retrieved if 0.3 < item['relevance'] <= 0.6][:1]
        for cr in creative:
            balanced_parts.append(f"Link: {cr['content']}")
            
        # Chaos (low relevance or chaotic)
        chaos = [item for item in retrieved if item['type'] == 'chaotic_memory'][:1]
        for ch in chaos:
            balanced_parts.append(f"Intuition: {ch['content']}")
            
        return " | ".join(balanced_parts)

class NicoleRAG:
    """Main Nicole RAG system"""
    
    def __init__(self, memory_db: str = "nicole_memory.db"):
        self.retriever = ChaoticRetriever(memory_db)
        self.augmenter = ContextAugmenter(self.retriever)
        self.rag_history = []
        self.adaptation_patterns = {}
        
    def generate_augmented_response(self, user_input: str, base_response: str = "",
                                  strategy: str = 'balanced') -> Tuple[str, str]:
        """Generates augmented response"""
        
        # Get augmented context
        augmented_context = self.augmenter.augment_context(user_input, strategy=strategy)
        
        # Analyze context to improve response
        improved_response = self._improve_response_with_context(
            user_input, base_response, augmented_context
        )
        
        # Save to RAG history
        self.rag_history.append({
            'user_input': user_input,
            'base_response': base_response,
            'improved_response': improved_response,
            'context': augmented_context,
            'strategy': strategy,
            'timestamp': time.time()
        })
        
        return improved_response, augmented_context
        
    def _improve_response_with_context(self, user_input: str, base_response: str,
                                     context: str) -> str:
        """Improves response based on context (NO TEMPLATES - only structural changes!)"""
        if not context or not base_response:
            # ANTI-TEMPLATE: just return base or None, let upper level decide
            return base_response

        # Structural improvements WITHOUT specific phrases:
        # If base_response already includes context - leave as is
        # Otherwise just return base_response without modifications
        # Real improvement should go through resonance, not templates!

        return base_response
        
    def adapt_retrieval_strategy(self, feedback_score: float, last_strategy: str):
        """Adapts search strategy based on feedback"""
        if last_strategy not in self.adaptation_patterns:
            self.adaptation_patterns[last_strategy] = {'scores': [], 'usage_count': 0}
            
        self.adaptation_patterns[last_strategy]['scores'].append(feedback_score)
        self.adaptation_patterns[last_strategy]['usage_count'] += 1
        
        # Adapt chaos factor
        if feedback_score > 0.7:  # Good result
            if last_strategy == 'chaotic':
                self.retriever.chaos_factor = min(0.3, self.retriever.chaos_factor * 1.1)
        elif feedback_score < 0.3:  # Bad result
            if last_strategy == 'chaotic':
                self.retriever.chaos_factor = max(0.05, self.retriever.chaos_factor * 0.9)
                
    def get_best_strategy(self) -> str:
        """Returns best strategy based on history"""
        if not self.adaptation_patterns:
            return 'balanced'
            
        strategy_scores = {}
        for strategy, data in self.adaptation_patterns.items():
            if data['scores']:
                avg_score = sum(data['scores']) / len(data['scores'])
                strategy_scores[strategy] = avg_score
                
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        else:
            return 'balanced'
            
    def get_rag_statistics(self) -> Dict:
        """RAG system statistics"""
        if not self.rag_history:
            return {'total_queries': 0}
            
        recent_history = self.rag_history[-100:]  # Last 100 queries
        
        strategies_used = Counter(item['strategy'] for item in recent_history)
        
        return {
            'total_queries': len(self.rag_history),
            'recent_queries': len(recent_history),
            'strategies_used': dict(strategies_used),
            'chaos_factor': self.retriever.chaos_factor,
            'adaptation_patterns': len(self.adaptation_patterns)
        }

# Global instance
nicole_rag = NicoleRAG()

def test_rag_system():
    """RAG system testing"""
    print("=== NICOLE RAG SYSTEM TEST ===")
    
    # Test 1: Basic search
    print("\\n--- Basic Search Test ---")
    context_results = nicole_rag.retriever.retrieve_context("programming work")
    print(f"Found {len(context_results)} context records")
    for result in context_results:
        print(f"- {result['type']}: {result['content'][:80]}... (relevance: {result['relevance']:.3f})")

    # Test 2: Different strategies
    print("\\n--- Strategies Test ---")
    strategies = ['factual', 'creative', 'chaotic', 'balanced']

    for strategy in strategies:
        response, context = nicole_rag.generate_augmented_response(
            "Tell me about programming",
            "Programming is cool",
            strategy=strategy
        )
        print(f"Strategy {strategy}: {response}")

    # Test 3: Adaptation
    print("\\n--- Adaptation Test ---")

    # Simulate feedback
    nicole_rag.adapt_retrieval_strategy(0.8, 'balanced')
    nicole_rag.adapt_retrieval_strategy(0.3, 'chaotic')
    nicole_rag.adapt_retrieval_strategy(0.9, 'creative')

    best_strategy = nicole_rag.get_best_strategy()
    print(f"Best strategy: {best_strategy}")

    # Statistics
    stats = nicole_rag.get_rag_statistics()
    print(f"\\nRAG Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
        
    print("\\n=== RAG TEST COMPLETED ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rag_system()
    else:
        print("Nicole RAG System ready to work")
        print("For testing run: python3 nicole_rag.py test")