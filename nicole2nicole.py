#!/usr/bin/env python3
"""
Nicole2Nicole - Weightless retraining module
Asynchronous retraining on transformer death/rebirth.
Learns from conversation logs and architecture evolution.
"""

import sqlite3
import json
import time
import threading
import math
import random
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import h2o
# import nicole  # Remove circular import

@dataclass
class LearningPattern:
    """Learning pattern from history"""
    input_pattern: str
    output_pattern: str
    metrics_context: Dict
    architecture_context: Dict
    success_score: float
    frequency: int = 1
    
class Nicole2NicoleCore:
    """Core retraining system"""

    def __init__(self, memory_db: str = "nicole_memory.db", knowledge_file: str = "nicole_learned_knowledge.json"):
        self.memory_db = memory_db
        self.knowledge_file = knowledge_file  # OPTIMIZATION: Auto-save file
        self.learning_patterns = {}
        self.evolution_patterns = {}
        self.architecture_preferences = {}
        self.learning_lock = threading.Lock()
        self.is_learning = False
        self.learning_cycles = 0  # OPTIMIZATION: Cycle counter for auto-save

        # OPTIMIZATION: Auto-load knowledge on startup
        if os.path.exists(self.knowledge_file):
            print(f"[Nicole2Nicole] Loading previous knowledge from {self.knowledge_file}...")
            self.import_learned_knowledge(self.knowledge_file)
        
    def analyze_conversation_logs(self, limit: int = 1000) -> List[LearningPattern]:
        """Analyzes conversation logs to extract patterns"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT user_input, nicole_output, metrics, transformer_config, timestamp
        FROM conversations 
        ORDER BY timestamp DESC 
        LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            user_input, nicole_output, metrics_str, config_str, timestamp = row
            
            try:
                metrics = json.loads(metrics_str)
                config = json.loads(config_str)


                # Create learning pattern
                pattern = LearningPattern(
                    input_pattern=self._extract_input_pattern(user_input),
                    output_pattern=self._extract_output_pattern(nicole_output),
                    metrics_context=metrics,
                    architecture_context=config,
                    success_score=self._calculate_success_score(metrics)
                )
                
                patterns.append(pattern)

            except Exception as e:
                print(f"[Nicole2Nicole] Log analysis error: {e}")
                
        return patterns
        
    def _extract_input_pattern(self, user_input: str) -> str:
        """Extracts pattern from user input"""
        words = user_input.lower().split()

        # Categorize message types
        if any(word in words for word in ['hello', 'hi', 'hey', 'greetings']):
            return "greeting"
        elif any(word in words for word in ['bye', 'goodbye', 'farewell']):
            return "farewell"
        elif any(word in words for word in ['how', 'what', 'doing']):
            return "status_inquiry"
        elif any(word in words for word in ['tell', 'explain', 'describe']):
            return "information_request"
        elif any(word in words for word in ['why', 'reason']):
            return "reasoning_request"
        elif len(words) > 10:
            return "long_message"
        elif len(words) < 3:
            return "short_message"
        else:
            return "general_conversation"
            
    def _extract_output_pattern(self, nicole_output: str) -> str:
        """Extracts pattern from Nicole's response via semantic analysis (NO TEMPLATES!)"""
        words = nicole_output.lower().split()

        # Analyze length and structure (not content!)
        word_count = len(words)
        has_question = '?' in nicole_output

        if word_count > 15:
            return "detailed_response"
        elif has_question:
            return "question_back"
        elif word_count < 5:
            return "brief_response"
        else:
            return "general_response"
            
    def _calculate_success_score(self, metrics: Dict) -> float:
        """Calculates interaction success score"""
        try:
            entropy = metrics.get('entropy', 0.5)
            perplexity = metrics.get('perplexity', 1.0)
            resonance = metrics.get('resonance', 0.3)
            coherence = metrics.get('coherence', 0.5)
            engagement = metrics.get('engagement', 0.5)

            # Weighted success score
            score = (
                entropy * 0.2 +
                (1.0 / max(0.1, perplexity)) * 0.3 +
                resonance * 0.3 +
                coherence * 0.1 +
                engagement * 0.1
            )

            return min(1.0, max(0.0, score))

        except Exception:
            return 0.5  # Neutral score on error
            
    def learn_from_patterns(self, patterns: List[LearningPattern]):
        """Learns from interaction patterns"""
        with self.learning_lock:
            self.is_learning = True

            try:
                # Group patterns by type
                pattern_groups = {}
                for pattern in patterns:
                    key = (pattern.input_pattern, pattern.output_pattern)
                    if key not in pattern_groups:
                        pattern_groups[key] = []
                    pattern_groups[key].append(pattern)

                # Analyze each group
                for (input_type, output_type), group in pattern_groups.items():
                    self._analyze_pattern_group(input_type, output_type, group)

                # Analyze architecture evolution
                self._analyze_architecture_evolution(patterns)

                print(f"[Nicole2Nicole] Learning completed on {len(patterns)} patterns")

            finally:
                self.is_learning = False
                
    def _analyze_pattern_group(self, input_type: str, output_type: str, patterns: List[LearningPattern]):
        """Analyzes group of similar patterns"""
        if len(patterns) < 2:
            return

        # Calculate average metrics for this interaction type
        avg_metrics = {}
        metric_keys = ['entropy', 'perplexity', 'resonance', 'coherence', 'engagement']

        for key in metric_keys:
            values = []
            for pattern in patterns:
                if key in pattern.metrics_context:
                    values.append(pattern.metrics_context[key])
            if values:
                avg_metrics[key] = sum(values) / len(values)

        # Find best architectures for this type
        best_patterns = sorted(patterns, key=lambda p: p.success_score, reverse=True)[:3]
        best_architectures = [p.architecture_context for p in best_patterns]

        # Save pattern
        pattern_key = f"{input_type}::{output_type}"
        self.learning_patterns[pattern_key] = {
            'avg_metrics': avg_metrics,
            'best_architectures': best_architectures,
            'frequency': len(patterns),
            'avg_success': sum(p.success_score for p in patterns) / len(patterns)
        }

        print(f"[Nicole2Nicole] Pattern {pattern_key}: {len(patterns)} examples, success {self.learning_patterns[pattern_key]['avg_success']:.3f}")
        
    def _analyze_architecture_evolution(self, patterns: List[LearningPattern]):
        """Analyzes architecture evolution"""
        # Group by architecture parameters
        arch_performance = {}

        for pattern in patterns:
            arch = pattern.architecture_context
            for param, value in arch.items():
                if param not in arch_performance:
                    arch_performance[param] = []
                arch_performance[param].append((value, pattern.success_score))

        # Find optimal ranges for each parameter
        for param, value_scores in arch_performance.items():
            if len(value_scores) < 5:
                continue

            # Sort by success
            sorted_values = sorted(value_scores, key=lambda x: x[1], reverse=True)
            top_20_percent = sorted_values[:max(1, len(sorted_values) // 5)]

            # Find range of best values
            top_values = [v[0] for v in top_20_percent]

            if isinstance(top_values[0], (int, float)):
                min_val = min(top_values)
                max_val = max(top_values)
                avg_val = sum(top_values) / len(top_values)

                self.architecture_preferences[param] = {
                    'min': min_val,
                    'max': max_val,
                    'avg': avg_val,
                    'samples': len(top_values)
                }

        print(f"[Nicole2Nicole] Analyzed architecture preferences for {len(self.architecture_preferences)} parameters")
        
    def suggest_architecture_improvements(self, current_arch: Dict, conversation_context: str) -> Dict:
        """
        Suggests architecture improvements based on learning + exploration noise

        ANTI-OVERFITTING: adds 10% chance of random exploration
        """
        if not self.learning_patterns or not self.architecture_preferences:
            return current_arch

        improved_arch = current_arch.copy()

        # EXPLORATION NOISE: 10% chance of random exploration
        # Prevents getting stuck in local optimum
        exploration_probability = 0.1

        if random.random() < exploration_probability:
            # Randomly select parameter to explore
            explorable_params = [p for p in improved_arch.keys()
                               if isinstance(improved_arch[p], (int, float))]

            if explorable_params:
                param_to_explore = random.choice(explorable_params)
                current_value = improved_arch[param_to_explore]

                # Random perturbation ±20%
                noise_factor = random.uniform(0.8, 1.2)
                improved_arch[param_to_explore] = current_value * noise_factor

                print(f"[Nicole2Nicole:Exploration] 🎲 Exploring parameter '{param_to_explore}': "
                      f"{current_value:.3f} → {improved_arch[param_to_explore]:.3f}")

        # Apply learned preferences
        for param, preferences in self.architecture_preferences.items():
            if param in improved_arch:
                current_val = improved_arch[param]

                # Move toward optimal range
                if isinstance(current_val, (int, float)):
                    target_val = preferences['avg']

                    # Smooth movement toward target
                    if current_val < preferences['min']:
                        improved_arch[param] = min(preferences['max'], current_val * 1.1)
                    elif current_val > preferences['max']:
                        improved_arch[param] = max(preferences['min'], current_val * 0.9)
                    else:
                        # Small random changes in optimal range
                        noise = random.uniform(-0.05, 0.05)
                        improved_arch[param] = current_val * (1 + noise)
                        
        return improved_arch
        
    def suggest_response_strategy(self, user_input: str, current_metrics: Dict) -> str:
        """Suggests response strategy based on learned patterns"""
        input_pattern = self._extract_input_pattern(user_input)

        # Find matching patterns
        matching_patterns = []
        for pattern_key, pattern_data in self.learning_patterns.items():
            if pattern_key.startswith(input_pattern + "::"):
                matching_patterns.append((pattern_key, pattern_data))

        if not matching_patterns:
            return "general_response"

        # Select best pattern
        best_pattern = max(matching_patterns, key=lambda x: x[1]['avg_success'])
        output_pattern = best_pattern[0].split("::")[-1]

        return output_pattern
        
    def continuous_learning_loop(self):
        """
        Continuous learning in background
        OPTIMIZATION: Auto-save every 10 cycles (~5 minutes)
        """
        while True:
            try:
                if not self.is_learning:
                    # Analyze new logs every 30 seconds
                    patterns = self.analyze_conversation_logs(100)
                    if patterns:
                        self.learn_from_patterns(patterns)
                        self.learning_cycles += 1

                        # OPTIMIZATION: Auto-save every 10 cycles
                        if self.learning_cycles % 10 == 0:
                            print(f"[Nicole2Nicole] Auto-saving knowledge (cycle {self.learning_cycles})...")
                            self.export_learned_knowledge(self.knowledge_file)

                time.sleep(30)

            except Exception as e:
                print(f"[Nicole2Nicole:ERROR] Continuous learning error: {e}")
                time.sleep(60)  # Increase interval on error

    def start_continuous_learning(self):
        """Starts continuous learning in separate thread"""
        learning_thread = threading.Thread(target=self.continuous_learning_loop, daemon=True)
        learning_thread.start()
        print("[Nicole2Nicole] Continuous learning started")

    def get_learning_statistics(self) -> Dict:
        """Returns learning statistics"""
        return {
            'learned_patterns': len(self.learning_patterns),
            'architecture_preferences': len(self.architecture_preferences),
            'is_learning': self.is_learning,
            'top_patterns': sorted(
                [(k, v['avg_success'], v['frequency']) for k, v in self.learning_patterns.items()],
                key=lambda x: x[1] * x[2],
                reverse=True
            )[:10]
        }
        
    def force_learning_session(self):
        """Forcibly starts learning session"""
        print("[Nicole2Nicole] Forced learning...")
        patterns = self.analyze_conversation_logs(500)
        if patterns:
            self.learn_from_patterns(patterns)
            return True
        return False

    def export_learned_knowledge(self, filepath: str):
        """Exports learned knowledge to file"""
        knowledge = {
            'learning_patterns': self.learning_patterns,
            'evolution_patterns': self.evolution_patterns,
            'architecture_preferences': self.architecture_preferences,
            'export_timestamp': time.time()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"[Nicole2Nicole] Knowledge exported to {filepath}")

    def import_learned_knowledge(self, filepath: str):
        """Imports learned knowledge from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)

            self.learning_patterns.update(knowledge.get('learning_patterns', {}))
            self.evolution_patterns.update(knowledge.get('evolution_patterns', {}))
            self.architecture_preferences.update(knowledge.get('architecture_preferences', {}))

            print(f"[Nicole2Nicole] Knowledge imported from {filepath}")
            return True

        except Exception as e:
            print(f"[Nicole2Nicole:ERROR] Import error: {e}")
            return False

# Integration with main Nicole system
class EnhancedFluidTransformer:  # Remove inheritance from nicole.FluidTransformer
    """Enhanced transformer with retraining"""

    def __init__(self, transformer_id: str, session_context: Dict = None, learning_core = None):
        # Remove super() call
        self.transformer_id = transformer_id
        self.session_context = session_context or {}
        self.learning_core = learning_core

        # Apply learned architecture improvements
        if self.learning_core:
            improved_arch = self.learning_core.suggest_architecture_improvements(
                self.architecture,
                session_context.get('conversation_context', '')
            )
            self.architecture = improved_arch

    def evolve_architecture(self, metrics):
        """Evolution with learned patterns"""
        # First standard evolution
        evolved = super().evolve_architecture(metrics)

        # Then apply learned improvements
        if self.learning_core and evolved:
            learned_improvements = self.learning_core.suggest_architecture_improvements(
                self.architecture,
                ""  # Conversation context
            )

            # Softly apply learned improvements
            for param, target_val in learned_improvements.items():
                if param in self.architecture and isinstance(target_val, (int, float)):
                    current_val = self.architecture[param]
                    # Move 10% toward learned optimum
                    self.architecture[param] = current_val * 0.9 + target_val * 0.1

        return evolved
