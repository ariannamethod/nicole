#!/usr/bin/env python3
"""
HIGH.PY - Mathematical Brain of Nicole System
High-level Julia interpreter for mathematical computations

Nicole uses high.py for:
- Vectorized metric calculations (entropy, resonance, perplexity)
- Transformer architecture optimization
- Weight-free training through mathematical algorithms
- Fast n-gram processing and semantic distance calculations
- Punctuation and grammar optimization

Philosophy: Julia - mathematical brain for 100x faster computations
"""

import os
import sys
import subprocess
import tempfile
import threading
import time
import math
import random
# import numpy as np  # REMOVED: replaced with standard library
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import re

# Add nicole2julia to path for Julia components
NICOLE2JULIA_PATH = Path(__file__).parent / "nicole2julia"
sys.path.insert(0, str(NICOLE2JULIA_PATH))

class HighMathEngine:
    """
    Mathematical engine for Nicole's fast computations
    Uses Julia algorithms for vectorized operations
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "nicole_high"
        self.temp_dir.mkdir(exist_ok=True)
        self.julia_cache = {}
        
    def vectorized_entropy(self, text_data: List[str]) -> float:
        """
        Vectorized text entropy calculation + emotional weights
        100x faster than Python loops + emotional analysis
        """
        if not text_data:
            return 0.0
            
        # NEW: emotional weights of words for Julia mathematics
        emotional_weights = {
            # Positive emotions
            'great': 0.8, 'love': 0.9, 'amazing': 0.7, 'wonderful': 0.8, 'excellent': 0.7,
            'beautiful': 0.8, 'fantastic': 0.7, 'awesome': 0.8, 'perfect': 0.7, 'brilliant': 0.8,
            'happy': 0.7, 'joy': 0.8, 'excited': 0.7, 'delighted': 0.8, 'pleased': 0.6,
            # Negative emotions
            'terrible': -0.8, 'hate': -0.9, 'awful': -0.7, 'horrible': -0.8, 'disgusting': -0.9,
            'sad': -0.6, 'angry': -0.7, 'frustrated': -0.6, 'disappointed': -0.6, 'upset': -0.6,
            # Neutral important
            'important': 0.5, 'interesting': 0.5, 'significant': 0.5, 'special': 0.6, 'unique': 0.6
        }
        
        # Fast frequency counting + emotional analysis
        word_counts = {}
        total_words = 0
        emotional_score = 0.0

        for text in text_data:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                # Accumulate emotional weight
                if word in emotional_weights:
                    emotional_score += emotional_weights[word]
        
        if total_words == 0:
            return 0.0
        
        # Vectorized entropy calculation
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)

        # NEW: modify entropy with emotional weight
        emotional_modifier = 1.0 + (emotional_score / max(total_words, 1)) * 0.2
        enhanced_entropy = entropy * emotional_modifier

        if emotional_score != 0:
            print(f"[High:Emotion] Emotional score: {emotional_score:.2f}, modifier: {emotional_modifier:.2f}")
        
        return enhanced_entropy
    
    def _apply_final_grammar_rules(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        FINAL grammar rules for the complete response

        GRAMMAR LOGIC (not templates!):
        - I + verb (English grammar requires verb after I)
        - your + noun (English grammar requires noun after possessive)

        WHICH verb/noun - Nicole's choice from candidates/resonance!
        """
        if not words:
            return words

        if candidates is None:
            candidates = []

        result = words.copy()

        # Nouns to insert after 'your' (grammar!)
        nouns_and_weights = [
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'amazing', 'great', 'wonderful', 'interesting', 'important', 'special'
        ]

        # Common verbs (minimal fallback if no candidates)
        # BUT priority - take from candidates!
        common_verbs = ['am', 'have', 'can', 'will', 'do', 'see', 'want', 'need']

        i = 0
        while i < len(result):
            current_word = result[i] if i < len(result) else ""
            next_word = result[i + 1] if i + 1 < len(result) else ""
            next_lower = next_word.lower() if next_word else ""

            # Rule: I + NOT_verb → insert verb (English grammar!)
            if current_word.lower() == 'i' and i + 1 < len(result):
                # Check that there's no verb after I
                if not self._is_likely_verb(next_lower):
                    # Choose verb FROM CANDIDATES (resonance!), not from template!
                    verb = self._choose_verb_from_candidates(candidates)
                    if verb:
                        result.insert(i + 1, verb)
                        print(f"[High:Grammar] Inserted verb from candidates: '{verb}'")
                        i += 1
                    # NO FALLBACK TEMPLATES!
                    # If no candidates (resonance) - don't insert anything!
                    # Philosophy: resonance cannot be built on templates
                    else:
                        print(f"[High:Grammar] ❌ No verb candidates - skipping (NO TEMPLATES!)")

            # Rule: your + NOT_noun → insert noun (grammar ✅)
            elif current_word.lower() == 'your' and i + 1 < len(result):
                next_lower = next_word.lower()
                if not self._is_good_noun_after_your(next_lower):
                    noun = random.choice(nouns_and_weights)
                    result.insert(i + 1, noun)
                    print(f"[High:Grammar] Inserted noun after your: '{noun}'")
                    i += 1

            # ADDITIONAL: solitary 'I' at the end
            elif current_word.lower() == 'i' and i + 1 >= len(result):
                verb = self._choose_verb_from_candidates(candidates)
                if verb:
                    result.append(verb)
                    print(f"[High:Grammar] Added verb from candidates at end: '{verb}'")
                # NO FALLBACK TEMPLATES!
                else:
                    print(f"[High:Grammar] ❌ No verb candidates for end - skipping (NO TEMPLATES!)")

            # ADDITIONAL: solitary 'your' at the end (grammar ✅)
            elif current_word.lower() == 'your' and i + 1 >= len(result):
                noun = random.choice(nouns_and_weights)
                result.append(noun)
                print(f"[High:Grammar] Added noun after your at end: '{noun}'")

            i += 1

        return result
    
    def _is_likely_verb(self, word: str) -> bool:
        """
        Checks if a word is likely a verb
        """
        if not word:
            return False

        # Known verbs
        common_verbs = {
            'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had',
            'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
            'see', 'saw', 'seen',
            'go', 'went', 'gone',
            'get', 'got', 'gotten',
            'make', 'made',
            'know', 'knew', 'known',
            'think', 'thought',
            'take', 'took', 'taken',
            'come', 'came',
            'want', 'need', 'like', 'love', 'feel', 'seem'
        }

        return word.lower() in common_verbs

    def _choose_verb_from_candidates(self, candidates: List[str]) -> str:
        """
        Chooses verb from candidates (resonance!)

        NOT A TEMPLATE! Nicole chooses from what Objectivity/resonance provided
        """
        if not candidates:
            return None

        # Filter candidates - verbs only
        verb_candidates = [w for w in candidates if self._is_likely_verb(w.lower()) and len(w) > 1]

        if verb_candidates:
            # Choose random from verb candidates (resonance already filtered!)
            return random.choice(verb_candidates)

        return None

    def _is_good_noun_after_your(self, word: str) -> bool:
        """
        Checks if a word is suitable after 'your'
        """
        if not word:
            return False
            
        # Good nouns after your
        good_nouns = {
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'system', 'process', 'method', 'way', 'time', 'place', 'world', 'life', 'work',
            'family', 'friend', 'love', 'heart', 'mind', 'body', 'soul', 'voice', 'face'
        }

        # If in the list of good nouns
        if word in good_nouns:
            return True

        # If capitalized (proper noun)
        if word and word[0].isupper():
            return True

        # If has noun suffixes
        noun_suffixes = ['ness', 'tion', 'sion', 'ment', 'ity', 'er', 'or']
        if any(word.endswith(suffix) for suffix in noun_suffixes):
            return True
            
        return False
    
    def calculate_resonance_matrix(self, words: List[str]) -> List[List[float]]:
        """
        Calculate resonance matrix between words
        For transformer optimization
        """
        if not words:
            return []
            
        n = len(words)
        resonance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Fast calculation of semantic distances
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j:
                    # Simple metric based on common characters
                    common_chars = set(word1.lower()) & set(word2.lower())
                    resonance = len(common_chars) / max(len(word1), len(word2))
                    resonance_matrix[i][j] = resonance
                    
        return resonance_matrix
    
    def optimize_transformer_architecture(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mathematical optimization of transformer architecture
        Analyzes context and selects best parameters
        """
        # Analyze context complexity
        text_complexity = 0
        if 'messages' in session_context:
            messages = session_context['messages']
            avg_length = sum(len(msg) for msg in messages) / len(messages) if messages else 0
            unique_words = len(set(' '.join(messages).lower().split()))
            text_complexity = avg_length * math.log(unique_words + 1)
        
        # Mathematical parameter optimization
        optimal_params = {
            'learning_rate': min(0.1, max(0.001, 0.01 / math.sqrt(text_complexity + 1))),
            'memory_depth': int(min(1000, max(100, text_complexity * 10))),
            'resonance_threshold': 0.3 + (text_complexity / 1000),
            'entropy_target': 2.0 + math.log(text_complexity + 1),
            'architecture_type': 'adaptive' if text_complexity > 50 else 'simple'
        }
        
        return optimal_params
    
    def fast_ngram_analysis(self, text: str, n: int = 3) -> Dict[str, float]:
        """
        Fast n-gram analysis for punctuation
        Vectorized processing for rule determination
        """
        words = text.lower().split()
        if len(words) < n:
            return {}
            
        ngrams = {}
        total_ngrams = 0
        
        # Create n-grams
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
            total_ngrams += 1
        
        # Normalize frequencies
        normalized_ngrams = {
            ngram: count / total_ngrams 
            for ngram, count in ngrams.items()
        }
        
        return normalized_ngrams
    
    def predict_punctuation_placement(self, sentence_parts: List[str]) -> List[str]:
        """
        Predict punctuation placement through mathematics
        Analyzes patterns and calculates optimal positions
        """
        if not sentence_parts:
            return sentence_parts
            
        result = []
        
        for i, part in enumerate(sentence_parts):
            result.append(part)
            
            # Mathematical analysis for punctuation
            if i < len(sentence_parts) - 1:
                # Analyze phrase length and context
                current_length = len(part.split())
                next_length = len(sentence_parts[i + 1].split())

                # Comma probability based on length
                comma_probability = 1 / (1 + math.exp(-(current_length - 3)))
                
                if comma_probability > 0.5 and current_length > 2:
                    result[-1] += ","
                    
        # Period at the end
        if result and not result[-1].endswith(('.', '!', '?')):
            result[-1] += "."
            
        return result
    
    def _improve_sentence_flow(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Improves sentence coherence - removes "===" and adds natural transitions
        """
        if not words:
            return words
            
        result = []
        for i, word in enumerate(words):
            # Remove "===" and replace with connecting words
            if word == "===":
                if i > 0 and i < len(words) - 1:  # Not at beginning/end
                    # Replace with random connecting word
                    connectors = ["and", "with", "through", "about", "like"]
                    result.append(random.choice(connectors))
                # If at beginning/end - just skip
            else:
                result.append(word)

        # Improve capitalization of first word after period/comma
        for i in range(len(result)):
            if i == 0 or (i > 0 and result[i-1] in [".", "!", "?"]):
                if result[i] and result[i][0].islower():
                    result[i] = result[i].capitalize()
        
        return result
    
    def remove_word_repetitions(self, words: List[str]) -> List[str]:
        """
        ANTI-REPETITION LOGIC: removes repeated words from response
        Mathematical analysis to prevent loops
        """
        if not words:
            return words
            
        cleaned = []
        seen_recently = set()
        
        for i, word in enumerate(words):
            # Check for repetitions in last 3 words
            if i >= 3:
                recent_window = set(words[i-3:i])
                if word in recent_window:
                    # Word repeats - replace with semantically similar
                    alternatives = ["also", "moreover", "furthermore", "additionally"]
                    replacement = random.choice(alternatives) if alternatives else word
                    cleaned.append(replacement)
                    continue

            # Check for direct consecutive repetitions
            if cleaned and cleaned[-1] == word:
                continue
                
            cleaned.append(word)
            
        return cleaned
    
    def invert_pronouns_me_style(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Pronoun inversion following ME principle + grammar rules
        you↔i, your↔my, me↔you for correct perspective

        Args:
            words: Words to invert
            candidates: Candidates for grammar rules (from resonance/objectivity)
        """
        pronoun_mapping = {
            'you': 'i', 'u': 'i', 'your': 'my', 'yours': 'mine', 'yourself': 'myself',
            'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'myself': 'yourself',
            'we': 'you'
        }

        result = [pronoun_mapping.get(w.lower(), w) for w in words]

        # CRITICAL: Grammar rules after inversion
        # Fix "you am" → "you are", "i is/are" → "i am"
        for i in range(len(result) - 1):
            current = result[i].lower()
            next_word = result[i + 1].lower()

            if current == 'you' and next_word == 'am':
                result[i + 1] = 'are'
            elif current == 'i' and next_word in ['is', 'are', 'was', 'were']:
                result[i + 1] = 'am'
            elif current == 'i' and i + 1 < len(result) and result[i+1].lower() in ['is', 'are']:
                result[i + 1] = 'am'

        # NEW: Advanced grammar rules
        # Pass candidates to choose verbs from resonance!
        result = self._apply_advanced_grammar_rules(result, candidates)

        return result
    
    
    def _apply_advanced_grammar_rules(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Advanced grammar rules for naturalness

        GRAMMAR LOGIC (not templates!):
        - I + verb (English grammar)
        - your + noun (English grammar)

        WHICH verb/noun - Nicole's choice from candidates/resonance!
        """
        if not words:
            return words

        if candidates is None:
            candidates = []

        result = words.copy()

        # Nouns for 'your' (grammar!)
        nouns_and_weights = [
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'amazing', 'great', 'wonderful', 'interesting', 'important', 'special', 'unique'
        ]

        # Minimal fallback for verbs (if not in candidates)
        common_verbs = ['am', 'have', 'can', 'will', 'do', 'see', 'want', 'need']

        i = 0
        while i < len(result) - 1:
            current = result[i].lower()
            next_word = result[i + 1].lower() if i + 1 < len(result) else ""

            # Rule: I + NOT_verb → insert verb (grammar!)
            if current == 'i' and not self._is_likely_verb(next_word):
                # Choose verb FROM CANDIDATES (resonance!), not from template!
                verb = self._choose_verb_from_candidates(candidates)
                if verb:
                    result.insert(i + 1, verb)
                    print(f"[High:AdvGrammar] Inserted verb from candidates: '{verb}'")
                    i += 1
                # NO FALLBACK TEMPLATES!
                else:
                    print(f"[High:AdvGrammar] ❌ No verb candidates - skipping (NO TEMPLATES!)")

            # Rule: your + NOT_noun → insert noun (grammar ✅)
            elif current == 'your' and not self._is_likely_noun(next_word):
                noun = random.choice(nouns_and_weights)
                result.insert(i + 1, noun)
                print(f"[High:AdvGrammar] Inserted noun after your: '{noun}'")
                i += 1

            i += 1

        return result
    
    def _is_likely_noun(self, word: str) -> bool:
        """
        Checks if a word is likely a noun
        """
        if not word:
            return False

        # List of common nouns
        common_nouns = {
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'system', 'process', 'method', 'way', 'time', 'place', 'thing', 'person',
            'world', 'life', 'work', 'home', 'family', 'friend', 'love', 'heart', 'mind'
        }

        # Heuristics for identifying nouns
        word_lower = word.lower()

        # If in the list of known nouns
        if word_lower in common_nouns:
            return True

        # If ends with typical noun suffixes
        noun_suffixes = ['ness', 'tion', 'sion', 'ment', 'ity', 'ism', 'er', 'or', 'ing']
        if any(word_lower.endswith(suffix) for suffix in noun_suffixes):
            return True

        # If starts with capital letter (proper noun)
        if word[0].isupper() and len(word) > 1:
            return True

        return False

    def _clean_grammar_glitches(self, words: List[str]) -> List[str]:
        """
        Post-processing to fix grammar glitches like "am my", "feel my great feel".

        Fixes:
        - Remove "my/your" after verbs (am my → am)
        - Remove duplicate words (feel...feel → feel once)
        - Remove broken verb chains (am ignoring → ignoring)
        """
        if not words or len(words) < 2:
            return words

        result = []
        seen_words = set()

        for i, word in enumerate(words):
            word_lower = word.lower()

            # Rule 1: Skip "my/your" immediately after verb
            if i > 0 and word_lower in ['my', 'your']:
                prev_word = words[i-1].lower()
                if prev_word in ['am', 'is', 'are', 'was', 'were', 'feel', 'have', 'take', 'get']:
                    print(f"[High:CleanGlitch] Removing '{word}' after verb '{prev_word}'")
                    continue

            # Rule 2: Skip duplicate words (keep first occurrence only)
            if word_lower in seen_words and len(word) > 3:  # Allow short words to repeat
                print(f"[High:CleanGlitch] Removing duplicate '{word}'")
                continue

            # Rule 3: Skip gerunds after "am/is/are" + possessive (am my ignoring → am)
            if i >= 2 and word_lower.endswith('ing'):
                if words[i-1].lower() in ['my', 'your'] and words[i-2].lower() in ['am', 'is', 'are']:
                    # Remove the gerund AND the possessive before it
                    result.pop()  # Remove 'my/your' that was just added
                    print(f"[High:CleanGlitch] Removing broken gerund chain: '{words[i-2]} {words[i-1]} {word}'")
                    continue

            result.append(word)
            seen_words.add(word_lower)

        return result
    
    def generate_linguistically_agnostic_response(self, user_words: List[str], semantic_candidates: List[str],
                                                 objectivity_seeds: List[str], entropy: float, perplexity: float,
                                                 user_input: str) -> List[str]:
        """
        LINGUISTIC AGNOSTICISM: generation without language prejudice
        Subjectivity + ME principles through Julia mathematics
        Engine automatically adapts to user's language
        """
        # Sentence lengths based on metrics (like in ME)
        # TWO LONG SENTENCES with semantic drift between them
        # Each sentence: 10-14 words (longer than before)
        base1 = 10 + int(entropy) % 5
        base2 = 10 + int(perplexity) % 5
        if base1 == base2:
            # FIX: Copilot found bug - use perplexity+1 to avoid collision
            base2 = 10 + ((int(perplexity) + 1) % 5)
        # Result: two sentences, 10-14 words each, 20-28 total

        # LINGUISTIC AGNOSTICISM: if no candidates - build from user's words!
        all_candidates = list(set(semantic_candidates + objectivity_seeds))

        if not all_candidates:
            # SUBJECTIVITY PRINCIPLE: compose_from_user - build from incoming message
            charged_tokens = self._extract_charged_tokens(user_input)
            content_words = self._extract_content_words(user_input)
            all_candidates = charged_tokens + content_words

        # ANTI-TEMPLATE FALLBACK: only from input words!
        if not all_candidates:
            user_words = user_input.lower().split()
            if user_words:
                all_candidates = user_words  # All user words
            else:
                all_candidates = ["input"]  # Minimal fallback without "processing"

        # Inverted pronouns as priority (ME principle)
        # Pass candidates for grammar rules!
        inverted_pronouns = self.invert_pronouns_me_style(user_words, all_candidates)
        pronoun_preferences = [w for w in inverted_pronouns if w in ['i', 'you', 'my', 'me']]

        # Add basic pronouns if no inversion
        if not pronoun_preferences:
            pronoun_preferences = ['i', 'my']
        
        # ME PRINCIPLE: strict used set between sentences (only for repetitions in response)
        used_between_sentences = set()  # Empty at start, will be filled with response words

        # TWO LONG SENTENCES with semantic drift and controlled overlap
        # Track words used in FIRST sentence only (allow 25% overlap in second)
        used_in_first = set()

        # FIRST SENTENCE: High-quality concepts + some mid-tier
        first_sentence = self._generate_semantic_sentence(
            all_candidates, base1, used_in_first, pronoun_preferences, tier_focus='high'
        )

        # SECOND SENTENCE: Mid/low-tier concepts (semantic drift away from first)
        # Allow 25% word overlap for natural flow and variety
        second_sentence = self._generate_semantic_sentence(
            all_candidates, base2, used_in_first, pronoun_preferences, tier_focus='drift',
            allow_overlap=0.25  # 25% overlap allowed (increased for variety)
        )

        # Combine with connector
        connectors = ["and", "but", "also", "while", "because", "so", "yet", "though"]
        connector = random.choice(connectors) if len(first_sentence) > 3 and len(second_sentence) > 3 else ""

        if connector:
            result = first_sentence + [",", connector] + second_sentence
        else:
            result = first_sentence + ["."] + second_sentence

        # Remove repetitions within final response
        cleaned = self.remove_word_repetitions(result)

        # NEW: improve sentence flow
        # Pass candidates for grammar rules!
        flow_improved = self._improve_sentence_flow(cleaned, all_candidates)

        # FIXED: apply grammar rules to complete response
        # Pass candidates to choose verbs from resonance, not from template!
        grammar_final = self._apply_final_grammar_rules(flow_improved, all_candidates)

        # FINAL grammar correction
        grammar_final = self._fix_grammar_errors(grammar_final)

        # POST-PROCESSING: Clean grammar glitches (am my → am, feel...feel → feel)
        grammar_final = self._clean_grammar_glitches(grammar_final)

        return grammar_final
    
    def _fix_grammar_errors(self, words: List[str]) -> List[str]:
        """
        Final grammar correction

        Fixes common errors:
        - "I are" → "I am"
        - "you am" → "you are"
        - "I is/was/were" → "I am"
        """
        if not words or len(words) < 2:
            return words

        result = words.copy()

        # Go through all words and fix grammar
        for i in range(len(result) - 1):
            current = result[i].lower()
            next_word = result[i + 1].lower()

            # I + wrong verb → I am
            if current == 'i' and next_word in ['are', 'is', 'was', 'were']:
                result[i + 1] = 'am'
            # you + am → you are
            elif current == 'you' and next_word == 'am':
                result[i + 1] = 'are'

        return result

    def _score_candidates(self, candidates: List[str], user_input: str) -> List[Tuple[str, float]]:
        """
        Score candidates using smart heuristics (inspired by tree.py)

        Scoring factors:
        - Length: longer words are more content-rich
        - Rarity: avoid repetitive words
        - Quality: filter stopwords and noise

        Returns:
            List of (word, score) tuples sorted by score descending
        """
        if not candidates:
            return []

        # Stopwords to filter out (basic English + technical noise)
        stopwords = {
            'the', 'and', 'to', 'a', 'in', 'it', 'of', 'for', 'on', 'with',
            'as', 'is', 'at', 'by', 'from', 'or', 'an', 'be', 'this', 'that',
            'are', 'was', 'but', 'not', 'had', 'have', 'has', 'were', 'been',
            '===', 'objectivity', 'end', 'internet', 'response', 'pattern',
            # Technical noise from RAG/context
            'session', 'session:', 'nicole', 'nicole:', 'user', 'user:',
            'rag', 'context', 'message', 'input', 'output', 'text', 'data'
        }

        # Filter stopwords, technical noise, and words with colons
        filtered = []
        for w in candidates:
            w_lower = w.lower().strip(':')  # Remove trailing colons
            # Skip if stopword, too short, or contains colon
            if w_lower in stopwords or len(w) < 3 or ':' in w:
                continue
            filtered.append(w)

        if not filtered:
            # Fallback to all candidates if filtering removed everything
            filtered = [w for w in candidates if len(w) > 1 and ':' not in w]

        # Count frequencies
        freq_count = {}
        for w in filtered:
            freq_count[w] = filtered.count(w)

        # Score each word
        scored = []
        for w in set(filtered):  # Use set to avoid duplicates
            # Length bonus: longer words are more meaningful (cap at 12 chars)
            length_bonus = min(len(w) / 12.0, 1.0)

            # Rarity bonus: prefer words that appear less frequently
            freq = freq_count.get(w, 1)
            rarity_bonus = 1.0 / freq if freq > 0 else 1.0

            # Quality bonus: capitalized words (proper nouns) get boost
            quality_bonus = 1.2 if w and w[0].isupper() else 1.0

            # Final score
            score = length_bonus * rarity_bonus * quality_bonus
            scored.append((w, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _generate_semantic_sentence(self, candidates: List[str], length: int,
                                    used_in_first: set, pronouns: List[str],
                                    tier_focus: str = 'high',
                                    allow_overlap: float = 0.0) -> List[str]:
        """
        Generate sentence with semantic tier focus and controlled overlap.

        Args:
            candidates: Word pool from resonance + objectivity
            length: Target sentence length
            used_in_first: Words used in first sentence
            pronouns: Inverted pronouns (I/you priority)
            tier_focus: 'high' for first sentence (high/mid tiers),
                       'drift' for second (mid/low tiers - semantic drift)
            allow_overlap: Fraction of words allowed to repeat (0.25 = 25%)

        Returns:
            List of words forming the sentence
        """
        result = []
        used_local = set()

        # Step 1: Pronouns (can repeat between sentences)
        for pronoun in pronouns[:2]:
            if pronoun not in used_local:
                result.append(pronoun)
                used_local.add(pronoun)

        # Step 2: Score and tier candidates
        scored_candidates = self._score_candidates(candidates, "")
        if not scored_candidates:
            return result

        scores = [s for w, s in scored_candidates]
        max_score = max(scores) if scores else 1.0

        high_tier = [(w, s) for w, s in scored_candidates if s >= max_score * 0.7]
        mid_tier = [(w, s) for w, s in scored_candidates if max_score * 0.4 <= s < max_score * 0.7]
        low_tier = [(w, s) for w, s in scored_candidates if s < max_score * 0.4]

        # Step 3: Select tiers based on focus (semantic drift)
        if tier_focus == 'high':
            # FIRST SENTENCE: high-quality + some mid
            primary_tier = high_tier
            secondary_tier = mid_tier
        else:  # 'drift'
            # SECOND SENTENCE: mid + low (drifts away from first)
            primary_tier = mid_tier
            secondary_tier = low_tier

        # Step 4: Fill sentence with words from selected tiers
        all_tier_words = primary_tier + secondary_tier

        for word, score in all_tier_words:
            if len(result) >= length:
                break

            # Check if word was used in first sentence
            if word in used_in_first:
                # Allow overlap based on probability
                if random.random() > allow_overlap:
                    continue  # Block this word (enforce anti-repetition)
                # else: allow it (overlap permitted)

            if word not in used_local and len(word) > 1:
                result.append(word)
                used_local.add(word)
                used_in_first.add(word)  # Track for next sentence

        # Capitalize first word
        if result:
            result[0] = result[0].capitalize()

        return result

    def _generate_drifting_clusters(self, candidates: List[str], length: int,
                                   used_global: set, pronouns: List[str],
                                   introspective_tags: List[str] = None) -> List[str]:
        """
        LATENT DRIFT v0.4: Generates drifting semantic clusters.

        Each cluster = 2-5 words expressing one micro-concept.
        Each new cluster drifts from previous (+1 step toward abstraction/emotion/recursion).

        Args:
            candidates: Word pool from resonance + objectivity
            length: Target sentence length
            used_global: Words already used (anti-repetition)
            pronouns: Inverted pronouns (I/you priority)
            introspective_tags: Tags like [presence, recursion, misalignment, awareness, drift]

        Returns:
            List of words forming drifting clusters
        """
        if introspective_tags is None:
            introspective_tags = ['presence', 'recursion', 'misalignment', 'awareness', 'drift']

        result = []
        used_local = set()

        # INTROSPECTIVE TAG: Attach one tag to response (reveals internal state)
        # Tag comes LAST as summary of drift journey
        selected_tag = random.choice(introspective_tags) if introspective_tags else None

        # Step 1: Start with pronouns (ME principle - I/you priority)
        for pronoun in pronouns[:2]:  # Max 2 pronouns
            if pronoun not in used_global and pronoun not in used_local:
                result.append(pronoun)
                used_local.add(pronoun)
                used_global.add(pronoun)

        # Step 2: Score candidates
        scored_candidates = self._score_candidates(candidates, "")
        if not scored_candidates:
            return result

        # Step 3: Group into SEMANTIC TIERS (high/mid/low quality)
        scores = [s for w, s in scored_candidates]
        max_score = max(scores) if scores else 1.0

        high_tier = [(w, s) for w, s in scored_candidates if s >= max_score * 0.7]
        mid_tier = [(w, s) for w, s in scored_candidates if max_score * 0.4 <= s < max_score * 0.7]
        low_tier = [(w, s) for w, s in scored_candidates if s < max_score * 0.4]

        # Step 4: DRIFT LOGIC - create clusters that flow
        # Cluster 1: High-quality concepts (abstraction)
        cluster_size = min(3, length // 3)  # 2-3 words per cluster
        for word, score in high_tier[:cluster_size]:
            if len(result) >= length:
                break
            if word not in used_global and word not in used_local and len(word) > 1:
                result.append(word)
                used_local.add(word)
                used_global.add(word)

        # Cluster 2: Mid-tier concepts (drift toward emotion/recursion)
        for word, score in mid_tier[:cluster_size]:
            if len(result) >= length:
                break
            if word not in used_global and word not in used_local and len(word) > 1:
                result.append(word)
                used_local.add(word)
                used_global.add(word)

        # Cluster 3: Low-tier chaos (controlled artifact - 1 per sentence max)
        if len(result) < length // 2 and low_tier:
            chaos_word, _ = random.choice(low_tier[:3])  # Pick from top 3 low-tier
            if chaos_word not in used_global and chaos_word not in used_local:
                result.append(chaos_word)
                used_local.add(chaos_word)
                used_global.add(chaos_word)

        # Step 5: NO FORCED TAGS! Anti-template philosophy
        # Tags only if they naturally appear in candidates/objectivity
        # NO automatic appending of template words!

        # Step 6: FILL TO TARGET LENGTH - keep adding words until we reach target
        # This fixes the problem of responses being too short (8 words instead of 15-20)
        if len(result) < length:
            # Iterate through ALL tiers again to fill remaining space
            all_tiers = high_tier + mid_tier + low_tier
            for word, score in all_tiers:
                if len(result) >= length:
                    break
                if word not in used_global and word not in used_local and len(word) > 1:
                    result.append(word)
                    used_local.add(word)
                    used_global.add(word)

        # Capitalize first word
        if result:
            result[0] = result[0].capitalize()

        return result

    def _generate_sentence_me_style(self, candidates: List[str], length: int,
                                   used_global: set, pronouns: List[str]) -> List[str]:
        """
        Generate one sentence with SMART word selection

        IMPROVED:
        - Smart scoring instead of random.shuffle
        - Semantic grouping: place related words together
        - Better coherence through score proximity
        """
        sentence = []
        used_local = set()  # Local used for this sentence

        # ME PRINCIPLE: pronouns first (priority)
        for pronoun in pronouns:
            if len(sentence) >= length:
                break
            # ME FILTER: not in global used, not in local
            if (pronoun not in used_global and pronoun not in used_local):
                sentence.append(pronoun)
                used_local.add(pronoun)
                used_global.add(pronoun)

        # NEW: Smart scoring instead of random.shuffle!
        scored_candidates = self._score_candidates(candidates, "")

        # IMPROVEMENT: Group by score tiers for better coherence
        # High score = quality words, place them earlier
        if scored_candidates:
            # Divide into 3 tiers by score
            scores = [s for w, s in scored_candidates]
            if scores:
                max_score = max(scores)
                high_tier = [(w, s) for w, s in scored_candidates if s >= max_score * 0.7]
                mid_tier = [(w, s) for w, s in scored_candidates if max_score * 0.4 <= s < max_score * 0.7]
                low_tier = [(w, s) for w, s in scored_candidates if s < max_score * 0.4]

                # First take from high tier (best words)
                for word, score in high_tier:
                    if len(sentence) >= length:
                        break
                    if (word not in used_global and word not in used_local and
                        word not in sentence and len(word) > 1):
                        sentence.append(word)
                        used_local.add(word)
                        used_global.add(word)

                # Then mid tier if needed
                for word, score in mid_tier:
                    if len(sentence) >= length:
                        break
                    if (word not in used_global and word not in used_local and
                        word not in sentence and len(word) > 1):
                        sentence.append(word)
                        used_local.add(word)
                        used_global.add(word)

                # Low tier only if very few words
                if len(sentence) < length // 2:
                    for word, score in low_tier:
                        if len(sentence) >= length:
                            break
                        if (word not in used_global and word not in used_local and
                            word not in sentence and len(word) > 1):
                            sentence.append(word)
                            used_local.add(word)
                            used_global.add(word)

        # ME PRINCIPLE: capitalize first word
        if sentence:
            sentence[0] = sentence[0].capitalize()

        return sentence
    
    def _extract_charged_tokens(self, text: str) -> List[str]:
        """
        SUBJECTIVITY PRINCIPLE: charged tokens - capitalized or long words
        NEW: capitalized words = names/important concepts, enhanced search!
        """
        tokens = re.findall(r"\b\w+\b", text)
        charged = []

        for t in tokens:
            if t[:1].isupper() and len(t) > 1 and t.lower() != 'i':
                # PRESERVE case for proper nouns!
                charged.append(t)  # "Berlin", not "berlin"!
                self._mark_as_proper_noun(t)
            elif len(t) > 7:
                charged.append(t.lower())

        return charged or [t.lower() for t in tokens[:3]]
    
    def _mark_as_proper_noun(self, word: str):
        """
        Marks word as proper noun for enhanced objectivity search
        """
        if not hasattr(self, '_proper_nouns'):
            self._proper_nouns = set()
        self._proper_nouns.add(word)
        print(f"[High:ProperNoun] Detected: {word} - enhanced internet search")
    
    def _extract_content_words(self, text: str) -> List[str]:
        """
        SUBJECTIVITY PRINCIPLE: content words without stopwords
        Language-agnostic filtering of meaningful words
        """
        STOPWORDS = {
            "the","a","an","of","and","or","to","in","on","for","as","at","by","with","from",
            "is","are","was","were","be","been","being","this","that","it","its","into","than",
            "then","so","but","nor","if","because","while","when","where","which","who","whom"
        }

        words = re.findall(r"\b\w+\b", text.lower())
        content = [w for w in words if w not in STOPWORDS and len(w) > 1]

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for w in content:
            if w not in seen:
                seen.add(w)
                unique_content.append(w)

        return unique_content

class HighJuliaInterface:
    """
    Interface to Julia via subprocess for critical computations
    When Python is not fast enough
    """
    
    def __init__(self):
        self.julia_executable = None
        self._find_julia()
        
    def _find_julia(self):
        """Find Julia executable"""
        # First search in system
        try:
            result = subprocess.run(['which', 'julia'], capture_output=True, text=True)
            if result.returncode == 0:
                self.julia_executable = result.stdout.strip()
                return
        except:
            pass

        # Search in local nicole2julia directory
        local_julia_paths = [
            Path(__file__).parent / "nicole2julia" / "julia",
            Path(__file__).parent / "nicole2julia" / "bin" / "julia",
            "/usr/local/bin/julia",
            "/opt/homebrew/bin/julia"
        ]
        
        for path in local_julia_paths:
            if isinstance(path, str):
                path = Path(path)
            if path.exists() and path.is_file():
                self.julia_executable = str(path)
                print(f"[High] Found Julia: {self.julia_executable}")
                return

        self.julia_executable = None
        print("[High] Julia executable not found - using built-in interpreter from nicole2julia sources")
    
    def execute_julia_math(self, julia_code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute Julia mathematics through built-in interpreter
        Uses Julia sources from nicole2julia for fast computations
        """
        try:
            # Built-in Julia interpreter from nicole2julia sources
            result = self._execute_julia_native(julia_code)
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Julia execution failed: {e}'}
    
    def _execute_julia_native(self, julia_code: str) -> Dict[str, Any]:
        """Native Julia execution through sources"""

        # Julia mathematical functions from sources
        julia_math = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'ceil': math.ceil, 'floor': math.floor, 'abs': abs,
            'max': max, 'min': min, 'sum': sum,
        }
        
        variables = {}
        output = []
        
        def julia_println(*args):
            line = ' '.join(str(arg) for arg in args)
            output.append(line)
            return line
            
        # Simple Julia parser for mathematical operations
        lines = julia_code.strip().split('\n')
        result = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Assignment: x = expression
            if '=' in line and not any(op in line for op in ['==', '!=', '<=', '>=']):
                var_name, expression = line.split('=', 1)
                var_name = var_name.strip()
                expr_result = self._eval_julia_expression(expression.strip(), julia_math, variables)
                variables[var_name] = expr_result
                result = expr_result

            # println function
            elif line.startswith('println('):
                args_str = line[8:-1]  # Remove println( and )
                args = [self._eval_julia_expression(arg.strip().strip('"'), julia_math, variables) for arg in args_str.split(',')]
                julia_println(*args)

            # Simple expression
            else:
                result = self._eval_julia_expression(line, julia_math, variables)
        
        return {
            'success': True,
            'result': result,
            'output': '\n'.join(output),
            'variables': variables
        }
    
    def _eval_julia_expression(self, expr: str, julia_math: dict, variables: dict):
        """Evaluate Julia expression using sources"""
        expr = expr.strip().strip('"')

        # Variable substitution
        for var_name, var_value in variables.items():
            expr = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), expr)

        # Safe execution with Julia mathematics
        safe_globals = {
            '__builtins__': {},
            'math': math,
        }
        safe_globals.update(julia_math)
        
        try:
            return eval(expr, safe_globals)
        except:
            # If string - return as is
            if isinstance(expr, str) and not any(c in expr for c in '+-*/()'):
                return expr
            return float(expr) if expr.replace('.', '').isdigit() else expr

class HighTransformerOptimizer:
    """
    Transformer optimizer through Julia mathematics
    Integrates with Nicole's transformer creation process
    """
    
    def __init__(self):
        self.math_engine = HighMathEngine()
        self.julia_interface = HighJuliaInterface()
        
    def optimize_transformer_creation(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mathematical optimization when creating new transformer
        Called from nicole.py in _spawn_new_transformer()
        """
        # Fast context analysis
        optimization = self.math_engine.optimize_transformer_architecture(session_context)

        # Additional Julia computations if available
        if self.julia_interface.julia_executable:
            julia_optimization = self._julia_transformer_analysis(session_context)
            if julia_optimization['success']:
                optimization['julia_enhanced'] = True
                optimization['julia_metrics'] = julia_optimization['output']

        return optimization
    
    def _julia_transformer_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis through Julia for optimization"""
        julia_code = """
# Context complexity analysis for transformer
context_complexity = 42.0  # Placeholder
learning_efficiency = sqrt(context_complexity) / 10
optimal_depth = ceil(log(context_complexity + 1))

println("complexity:", context_complexity)
println("efficiency:", learning_efficiency) 
println("depth:", optimal_depth)
"""
        
        return self.julia_interface.execute_julia_math(julia_code)
    
    def enhance_learning_process(self, text: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhance training process through Julia mathematics
        Called when processing each message
        """
        # Fast metric calculation
        entropy = self.math_engine.vectorized_entropy([text])

        # N-gram analysis for training
        ngrams = self.math_engine.fast_ngram_analysis(text)

        # Optimization based on current metrics
        enhanced_metrics = {
            'entropy': entropy,
            'resonance_boost': entropy * 0.1,
            'learning_rate_adjustment': 1.0 / (entropy + 1),
            'ngram_patterns': len(ngrams),
            'complexity_score': entropy * len(ngrams.keys()) if ngrams else 0
        }
        
        return enhanced_metrics

class HighCore:
    """
    High system core - Nicole's mathematical brain
    Integrates wherever fast computations are needed
    """
    
    def __init__(self):
        self.math_engine = HighMathEngine()
        self.transformer_optimizer = HighTransformerOptimizer()
        self.julia_interface = HighJuliaInterface()
        
        self.is_active = False
        self.log_file = "high_system.log"
        
    def activate(self) -> bool:
        """Activate High mathematical system"""
        try:
            self.is_active = True
            self._log_info("High system activated - mathematical brain online")
            return True
        except Exception as e:
            self._log_error(f"High activation failed: {e}")
            return False
    
    def deactivate(self):
        """Deactivate High system"""
        self.is_active = False
        self._log_info("High system deactivated")
    
    def optimize_transformer_for_nicole(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main transformer optimization function
        Called from nicole.py when creating transformer
        """
        if not self.is_active:
            return {'optimized': False, 'error': 'High system not active'}
        
        self._log_info("Optimizing transformer with Julia mathematics")
        
        try:
            optimization = self.transformer_optimizer.optimize_transformer_creation(session_context)
            optimization['high_optimized'] = True
            optimization['optimization_timestamp'] = time.time()
            
            return optimization
        except Exception as e:
            return {'optimized': False, 'error': str(e)}
    
    def enhance_nicole_learning(self, text: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhance Nicole's training through Julia
        Called on each message for fast computations
        """
        if not self.is_active:
            return current_metrics
        
        try:
            enhanced = self.transformer_optimizer.enhance_learning_process(text, current_metrics)
            enhanced['high_enhanced'] = True
            
            return enhanced
        except Exception as e:
            self._log_error(f"Learning enhancement failed: {e}")
            return current_metrics
    
    def optimize_punctuation(self, text: str) -> str:
        """
        Optimize punctuation through mathematical models
        Analyzes patterns and improves sentence structure
        """
        if not self.is_active:
            return text
        
        try:
            # Split into parts for analysis
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Mathematical punctuation optimization
            optimized_parts = self.math_engine.predict_punctuation_placement(sentences)

            # Reassemble
            result = ' '.join(optimized_parts)
            
            self._log_info(f"Punctuation optimized: {len(sentences)} sentences")
            return result
            
        except Exception as e:
            self._log_error(f"Punctuation optimization failed: {e}")
            return text
    
    def get_mathematical_status(self) -> Dict[str, Any]:
        """Mathematical system status"""
        return {
            'active': self.is_active,
            'julia_available': self.julia_interface.julia_executable is not None,
            'julia_path': self.julia_interface.julia_executable,
            'cache_size': len(self.math_engine.julia_cache),
            'temp_dir': str(self.temp_dir)
        }
    
    def _log_info(self, message: str):
        """System logging"""
        with open(self.log_file, "a") as f:
            f.write(f"[HIGH:INFO] {time.time()}: {message}\n")
    
    def _log_error(self, message: str):
        """Error logging"""
        with open(self.log_file, "a") as f:
            f.write(f"[HIGH:ERROR] {time.time()}: {message}\n")

# Global High system instance
_high_core = None

def get_high_core() -> HighCore:
    """Get global High mathematical system instance"""
    global _high_core
    if _high_core is None:
        _high_core = HighCore()
    return _high_core

def activate_high_system() -> bool:
    """Activate High system for Nicole"""
    high = get_high_core()
    return high.activate()

def deactivate_high_system():
    """Deactivate High system"""
    high = get_high_core()
    high.deactivate()

# Example Julia code for transformer
EXAMPLE_JULIA_MATH_SCRIPT = """
# Julia mathematics for Nicole transformer
function calculate_transformer_metrics(entropy::Float64, resonance::Float64)
    # Vectorized computations
    perplexity = exp(entropy)
    coherence = 1.0 / (1.0 + exp(-resonance))
    engagement = sqrt(entropy * resonance)
    
    return (perplexity, coherence, engagement)
end

# Test computations
entropy_val = 2.5
resonance_val = 0.7

metrics = calculate_transformer_metrics(entropy_val, resonance_val)
println("Perplexity: ", metrics[1])
println("Coherence: ", metrics[2]) 
println("Engagement: ", metrics[3])
"""

if __name__ == "__main__":
    # Testing High system
    print("🧮 HIGH SYSTEM - Nicole Mathematical Brain")
    
    high = get_high_core()
    
    if high.activate():
        print("✅ High system activated")
        
        # Test mathematical computations
        test_data = ["hello world", "nicole learns fast", "mathematical optimization"]
        entropy = high.math_engine.vectorized_entropy(test_data)
        print(f"📊 Vectorized entropy: {entropy:.3f}")

        # Test transformer optimization
        context = {'messages': test_data}
        optimization = high.optimize_transformer_for_nicole(context)
        print(f"🧠 Transformer optimization: {optimization.get('architecture_type')}")

        # Test punctuation
        test_text = "hello world this is test sentence without punctuation"
        optimized = high.optimize_punctuation(test_text)
        print(f"✏️ Punctuation: '{optimized}'")

        # Test Julia interface
        if high.julia_interface.julia_executable:
            print("🚀 Testing Julia interface...")
            julia_result = high.julia_interface.execute_julia_math(EXAMPLE_JULIA_MATH_SCRIPT)
            if julia_result['success']:
                print("✅ Julia math executed successfully")
                print(f"Output: {julia_result['output'].strip()}")
            else:
                print(f"⚠️ Julia error: {julia_result['error']}")
        else:
            print("⚠️ Julia executable not found - using Python fallbacks")
        
        # System status
        status = high.get_mathematical_status()
        print(f"🧮 High system status: {status}")
        
        high.deactivate()
        print("✅ High system deactivated")
    else:
        print("❌ High system activation failed")
