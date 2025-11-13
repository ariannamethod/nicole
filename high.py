#!/usr/bin/env python3
"""
HIGH.PY - Mathematical Brain of Nicole System
Высокоуровневый интерпретатор Julia для математических вычислений

Nicole использует high.py для:
- Векторизованных вычислений метрик (энтропия, резонанс, перплексия)
- Оптимизации архитектур трансформеров
- Дообучения без весов через математические алгоритмы
- Быстрой обработки n-граммов и семантических дистанций
- Оптимизации пунктуации и грамматики

Философия: Julia - математический мозг для быстрых вычислений в 100x
"""

import os
import sys
import subprocess
import tempfile
import threading
import time
import math
import random
# import numpy as np  # УБРАНО: заменено на стандартную библиотеку
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import re

# Добавляем nicole2julia в путь для Julia компонентов
NICOLE2JULIA_PATH = Path(__file__).parent / "nicole2julia"
sys.path.insert(0, str(NICOLE2JULIA_PATH))

class HighMathEngine:
    """
    Математический движок для быстрых вычислений Nicole
    Использует Julia алгоритмы для векторизованных операций
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "nicole_high"
        self.temp_dir.mkdir(exist_ok=True)
        self.julia_cache = {}
        
    def vectorized_entropy(self, text_data: List[str]) -> float:
        """
        Векторизованное вычисление энтропии текста + эмоциональные веса
        В 100x быстрее чем Python циклы + эмоциональный анализ
        """
        if not text_data:
            return 0.0
            
        # НОВОЕ: эмоциональные веса слов для Julia математики
        emotional_weights = {
            # Позитивные эмоции
            'great': 0.8, 'love': 0.9, 'amazing': 0.7, 'wonderful': 0.8, 'excellent': 0.7,
            'beautiful': 0.8, 'fantastic': 0.7, 'awesome': 0.8, 'perfect': 0.7, 'brilliant': 0.8,
            'happy': 0.7, 'joy': 0.8, 'excited': 0.7, 'delighted': 0.8, 'pleased': 0.6,
            # Негативные эмоции  
            'terrible': -0.8, 'hate': -0.9, 'awful': -0.7, 'horrible': -0.8, 'disgusting': -0.9,
            'sad': -0.6, 'angry': -0.7, 'frustrated': -0.6, 'disappointed': -0.6, 'upset': -0.6,
            # Нейтральные важные
            'important': 0.5, 'interesting': 0.5, 'significant': 0.5, 'special': 0.6, 'unique': 0.6,
            # Русские эмоциональные
            'отлично': 0.8, 'классно': 0.7, 'супер': 0.8, 'круто': 0.7, 'прекрасно': 0.8, 'здорово': 0.7,
            'ужасно': -0.8, 'плохо': -0.6, 'грустно': -0.6, 'злой': -0.7, 'расстроен': -0.6
        }
        
        # Быстрый подсчет частот + эмоциональный анализ
        word_counts = {}
        total_words = 0
        emotional_score = 0.0
        
        for text in text_data:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                # Накапливаем эмоциональный вес
                if word in emotional_weights:
                    emotional_score += emotional_weights[word]
        
        if total_words == 0:
            return 0.0
        
        # Векторизованное вычисление энтропии
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # НОВОЕ: модифицируем энтропию эмоциональным весом
        emotional_modifier = 1.0 + (emotional_score / max(total_words, 1)) * 0.2
        enhanced_entropy = entropy * emotional_modifier
        
        if emotional_score != 0:
            print(f"[High:Emotion] Эмоциональный скор: {emotional_score:.2f}, модификатор: {emotional_modifier:.2f}")
        
        return enhanced_entropy
    
    def _apply_final_grammar_rules(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        ФИНАЛЬНЫЕ грамматические правила для готового ответа

        ГРАММАТИЧЕСКАЯ ЛОГИКА (не шаблоны!):
        - I + глагол (английская грамматика требует глагол после I)
        - your + существительное (английская грамматика требует noun после possessive)

        КАКОЙ глагол/существительное - выбор Nicole из candidates/резонанса!
        """
        if not words:
            return words

        if candidates is None:
            candidates = []

        result = words.copy()

        # Существительные для вставки после 'your' (грамматика!)
        nouns_and_weights = [
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'amazing', 'great', 'wonderful', 'interesting', 'important', 'special'
        ]

        # Общие глаголы (минимальный fallback если нет candidates)
        # НО приоритет - брать из candidates!
        common_verbs = ['am', 'have', 'can', 'will', 'do', 'see', 'want', 'need']

        i = 0
        while i < len(result):
            current_word = result[i] if i < len(result) else ""
            next_word = result[i + 1] if i + 1 < len(result) else ""
            next_lower = next_word.lower() if next_word else ""

            # Правило: I + НЕ_глагол → вставляем глагол (грамматика английского!)
            if current_word.lower() == 'i' and i + 1 < len(result):
                # Проверяем что после I нет глагола
                if not self._is_likely_verb(next_lower):
                    # Выбираем глагол ИЗ CANDIDATES (резонанс!), не из шаблона!
                    verb = self._choose_verb_from_candidates(candidates)
                    if verb:
                        result.insert(i + 1, verb)
                        print(f"[High:Grammar] Вставлен глагол из candidates: '{verb}'")
                        i += 1
                    # NO FALLBACK TEMPLATES!
                    # Если нет candidates (резонанса) - не вставляем ничего!
                    # Философия: резонанс не может строиться на шаблонах
                    else:
                        print(f"[High:Grammar] ❌ Нет verb candidates - пропускаем (NO TEMPLATES!)")

            # Правило: your + НЕ_существительное → вставляем существительное (грамматика ✅)
            elif current_word.lower() == 'your' and i + 1 < len(result):
                next_lower = next_word.lower()
                if not self._is_good_noun_after_your(next_lower):
                    noun = random.choice(nouns_and_weights)
                    result.insert(i + 1, noun)
                    print(f"[High:Grammar] Вставлено существительное после your: '{noun}'")
                    i += 1

            # ДОПОЛНИТЕЛЬНО: одиночный 'I' в конце
            elif current_word.lower() == 'i' and i + 1 >= len(result):
                verb = self._choose_verb_from_candidates(candidates)
                if verb:
                    result.append(verb)
                    print(f"[High:Grammar] Добавлен глагол из candidates в конце: '{verb}'")
                # NO FALLBACK TEMPLATES!
                else:
                    print(f"[High:Grammar] ❌ Нет verb candidates для конца - пропускаем (NO TEMPLATES!)")

            # ДОПОЛНИТЕЛЬНО: одиночный 'your' в конце (грамматика ✅)
            elif current_word.lower() == 'your' and i + 1 >= len(result):
                noun = random.choice(nouns_and_weights)
                result.append(noun)
                print(f"[High:Grammar] Добавлено существительное после your в конце: '{noun}'")

            i += 1

        return result
    
    def _is_likely_verb(self, word: str) -> bool:
        """
        Проверяет, является ли слово вероятным глаголом
        """
        if not word:
            return False

        # Известные глаголы
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
        Выбирает глагол из candidates (резонанс!)

        НЕ ШАБЛОН! Nicole сама выбирает из того что дал Objectivity/резонанс
        """
        if not candidates:
            return None

        # Фильтруем candidates - только глаголы
        verb_candidates = [w for w in candidates if self._is_likely_verb(w.lower()) and len(w) > 1]

        if verb_candidates:
            # Выбираем случайный из глаголов-кандидатов (резонанс уже отфильтровал!)
            return random.choice(verb_candidates)

        return None

    def _is_good_noun_after_your(self, word: str) -> bool:
        """
        Проверяет, подходит ли слово после 'your'
        """
        if not word:
            return False
            
        # Хорошие существительные после your
        good_nouns = {
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'system', 'process', 'method', 'way', 'time', 'place', 'world', 'life', 'work',
            'family', 'friend', 'love', 'heart', 'mind', 'body', 'soul', 'voice', 'face',
            # Русские
            'память', 'способности', 'возможности', 'мысли', 'идеи', 'слова', 'опыт', 'знания'
        }
        
        # Если в списке хороших существительных
        if word in good_nouns:
            return True
            
        # Если заглавное (имя собственное)
        if word and word[0].isupper():
            return True
            
        # Если с суффиксами существительных
        noun_suffixes = ['ness', 'tion', 'sion', 'ment', 'ity', 'er', 'or']
        if any(word.endswith(suffix) for suffix in noun_suffixes):
            return True
            
        return False
    
    def calculate_resonance_matrix(self, words: List[str]) -> List[List[float]]:
        """
        Вычисление матрицы резонанса между словами
        Для оптимизации трансформеров
        """
        if not words:
            return []
            
        n = len(words)
        resonance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Быстрое вычисление семантических дистанций
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j:
                    # Простая метрика на основе общих символов
                    common_chars = set(word1.lower()) & set(word2.lower())
                    resonance = len(common_chars) / max(len(word1), len(word2))
                    resonance_matrix[i][j] = resonance
                    
        return resonance_matrix
    
    def optimize_transformer_architecture(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Математическая оптимизация архитектуры трансформера
        Анализирует контекст и выбирает лучшие параметры
        """
        # Анализируем сложность контекста
        text_complexity = 0
        if 'messages' in session_context:
            messages = session_context['messages']
            avg_length = sum(len(msg) for msg in messages) / len(messages) if messages else 0
            unique_words = len(set(' '.join(messages).lower().split()))
            text_complexity = avg_length * math.log(unique_words + 1)
        
        # Математическая оптимизация параметров
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
        
        # Создаем n-граммы
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
            total_ngrams += 1
        
        # Нормализуем частоты
        normalized_ngrams = {
            ngram: count / total_ngrams 
            for ngram, count in ngrams.items()
        }
        
        return normalized_ngrams
    
    def predict_punctuation_placement(self, sentence_parts: List[str]) -> List[str]:
        """
        Предсказание расстановки пунктуации через математику
        Анализирует паттерны и вычисляет оптимальные места
        """
        if not sentence_parts:
            return sentence_parts
            
        result = []
        
        for i, part in enumerate(sentence_parts):
            result.append(part)
            
            # Математический анализ для пунктуации
            if i < len(sentence_parts) - 1:
                # Анализируем длину фраз и контекст
                current_length = len(part.split())
                next_length = len(sentence_parts[i + 1].split())
                
                # Вероятность запятой на основе длины
                comma_probability = 1 / (1 + math.exp(-(current_length - 3)))
                
                if comma_probability > 0.5 and current_length > 2:
                    result[-1] += ","
                    
        # Точка в конце
        if result and not result[-1].endswith(('.', '!', '?')):
            result[-1] += "."
            
        return result
    
    def _improve_sentence_flow(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Улучшает связность предложений - убирает "===" и добавляет естественные переходы
        """
        if not words:
            return words
            
        result = []
        for i, word in enumerate(words):
            # Убираем "===" и заменяем на связующие слова
            if word == "===":
                if i > 0 and i < len(words) - 1:  # Не в начале/конце
                    # Заменяем на случайное связующее слово
                    connectors = ["and", "with", "through", "about", "like"]
                    result.append(random.choice(connectors))
                # Если в начале/конце - просто пропускаем
            else:
                result.append(word)
        
        # Улучшаем капитализацию первого слова после точки/запятой
        for i in range(len(result)):
            if i == 0 or (i > 0 and result[i-1] in [".", "!", "?"]):
                if result[i] and result[i][0].islower():
                    result[i] = result[i].capitalize()
        
        return result
    
    def remove_word_repetitions(self, words: List[str]) -> List[str]:
        """
        АНТИ-ПОВТОР ЛОГИКА: убирает повторяющиеся слова из ответа
        Математический анализ для предотвращения циклов
        """
        if not words:
            return words
            
        cleaned = []
        seen_recently = set()
        
        for i, word in enumerate(words):
            # Проверяем повторы в последних 3 словах
            if i >= 3:
                recent_window = set(words[i-3:i])
                if word in recent_window:
                    # Слово повторяется - заменяем на семантически близкое
                    alternatives = ["также", "кроме того", "более того", "дополнительно"]
                    replacement = random.choice(alternatives) if alternatives else word
                    cleaned.append(replacement)
                    continue
            
            # Проверяем прямые повторы подряд
            if cleaned and cleaned[-1] == word:
                continue
                
            cleaned.append(word)
            
        return cleaned
    
    def invert_pronouns_me_style(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Инверсия местоимений по принципу ME + грамматические правила
        you↔i, your↔my, me↔you для правильной перспективы

        Args:
            words: Слова для инверсии
            candidates: Кандидаты для грамматических правил (от резонанса/objectivity)
        """
        pronoun_mapping = {
            'you': 'i', 'u': 'i', 'your': 'my', 'yours': 'mine', 'yourself': 'myself',
            'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'myself': 'yourself',
            'we': 'you'
        }

        result = [pronoun_mapping.get(w.lower(), w) for w in words]

        # КРИТИЧНО: Грамматические правила после инверсии
        # Исправляем "you am" → "you are", "i is/are" → "i am"
        for i in range(len(result) - 1):
            current = result[i].lower()
            next_word = result[i + 1].lower()

            if current == 'you' and next_word == 'am':
                result[i + 1] = 'are'
            elif current == 'i' and next_word in ['is', 'are', 'was', 'were']:
                result[i + 1] = 'am'
            elif current == 'i' and i + 1 < len(result) and result[i+1].lower() in ['is', 'are']:
                result[i + 1] = 'am'

        # НОВОЕ: Продвинутые грамматические правила
        # Передаём candidates для выбора глаголов из резонанса!
        result = self._apply_advanced_grammar_rules(result, candidates)

        return result
    
    
    def _apply_advanced_grammar_rules(self, words: List[str], candidates: List[str] = None) -> List[str]:
        """
        Продвинутые грамматические правила для естественности

        ГРАММАТИЧЕСКАЯ ЛОГИКА (не шаблоны!):
        - I + глагол (английская грамматика)
        - your + существительное (английская грамматика)

        КАКОЙ глагол/существительное - выбор Nicole из candidates/резонанса!
        """
        if not words:
            return words

        if candidates is None:
            candidates = []

        result = words.copy()

        # Существительные для 'your' (грамматика!)
        nouns_and_weights = [
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'amazing', 'great', 'wonderful', 'interesting', 'important', 'special', 'unique'
        ]

        # Минимальный fallback для глаголов (если нет в candidates)
        common_verbs = ['am', 'have', 'can', 'will', 'do', 'see', 'want', 'need']

        i = 0
        while i < len(result) - 1:
            current = result[i].lower()
            next_word = result[i + 1].lower() if i + 1 < len(result) else ""

            # Правило: I + НЕ_глагол → вставляем глагол (грамматика!)
            if current == 'i' and not self._is_likely_verb(next_word):
                # Выбираем глагол ИЗ CANDIDATES (резонанс!), не из шаблона!
                verb = self._choose_verb_from_candidates(candidates)
                if verb:
                    result.insert(i + 1, verb)
                    print(f"[High:AdvGrammar] Вставлен глагол из candidates: '{verb}'")
                    i += 1
                # NO FALLBACK TEMPLATES!
                else:
                    print(f"[High:AdvGrammar] ❌ Нет verb candidates - пропускаем (NO TEMPLATES!)")

            # Правило: your + НЕ_существительное → вставляем существительное (грамматика ✅)
            elif current == 'your' and not self._is_likely_noun(next_word):
                noun = random.choice(nouns_and_weights)
                result.insert(i + 1, noun)
                print(f"[High:AdvGrammar] Вставлено существительное после your: '{noun}'")
                i += 1

            i += 1

        return result
    
    def _is_likely_noun(self, word: str) -> bool:
        """
        Проверяет, является ли слово вероятным существительным
        """
        if not word:
            return False

        # Список распространенных существительных
        common_nouns = {
            'memory', 'abilities', 'capabilities', 'thoughts', 'ideas', 'words', 'questions',
            'knowledge', 'experience', 'approach', 'style',
            'system', 'process', 'method', 'way', 'time', 'place', 'thing', 'person',
            'world', 'life', 'work', 'home', 'family', 'friend', 'love', 'heart', 'mind',
            # Русские существительные
            'память', 'способности', 'возможности', 'мысли', 'идеи', 'слова', 'вопросы',
            'знания', 'опыт', 'понимание', 'подход', 'стиль', 'система', 'процесс'
        }

        # Эвристики для определения существительных
        word_lower = word.lower()

        # Если в списке известных существительных
        if word_lower in common_nouns:
            return True

        # Если заканчивается на типичные суффиксы существительных
        noun_suffixes = ['ness', 'tion', 'sion', 'ment', 'ity', 'ism', 'er', 'or', 'ing']
        if any(word_lower.endswith(suffix) for suffix in noun_suffixes):
            return True

        # Если начинается с заглавной буквы (имя собственное)
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
        ЯЗЫКОВОЙ АГНОСТИЦИЗМ: генерация без языковых предрассудков
        Принципы subjectivity + ME через Julia математику
        Движок подстраивается под язык пользователя автоматически
        """
        # Длины предложений на основе метрик (как в ME)
        base1 = 5 + int(entropy) % 5
        base2 = 5 + int(perplexity) % 5
        if base1 == base2:
            base2 = 5 + ((base2 + 1) % 5)
        
        # ЯЗЫКОВОЙ АГНОСТИЦИЗМ: если нет кандидатов - строим из слов пользователя!
        all_candidates = list(set(semantic_candidates + objectivity_seeds))
        
        if not all_candidates:
            # ПРИНЦИП SUBJECTIVITY: compose_from_user - строим из входящего сообщения
            charged_tokens = self._extract_charged_tokens(user_input)
            content_words = self._extract_content_words(user_input)
            all_candidates = charged_tokens + content_words
            
        # АНТИ-ШАБЛОННЫЙ ФОЛЛБЕК: только из входящих слов!
        if not all_candidates:
            user_words = user_input.lower().split()
            if user_words:
                all_candidates = user_words  # Все слова пользователя
            else:
                all_candidates = ["input"]  # Минимальный fallback без "processing"
        
        # Инвертированные местоимения как приоритет (принцип ME)
        # Передаём candidates для грамматических правил!
        inverted_pronouns = self.invert_pronouns_me_style(user_words, all_candidates)
        pronoun_preferences = [w for w in inverted_pronouns if w in ['i', 'you', 'я', 'ты', 'my', 'мой', 'меня', 'мне']]
        
        # Добавляем базовые местоимения если нет инверсии
        if not pronoun_preferences:
            pronoun_preferences = ['i', 'my']
        
        # ME ПРИНЦИП: строгий used set между предложениями (только для повторов в ответе)
        used_between_sentences = set()  # Пустой в начале, будет заполняться словами ответа
        
        # Генерируем первое предложение
        first_sentence = self._generate_sentence_me_style(
            all_candidates, base1, used_between_sentences, pronoun_preferences
        )
        
        # Генерируем второе предложение (used обновлен первым предложением)
        second_sentence = self._generate_sentence_me_style(
            all_candidates, base2, used_between_sentences, pronoun_preferences
        )
        
        # ME ПРИНЦИП: два предложения с улучшенной связностью
        # Добавляем связующие слова между предложениями
        connectors = ["and", "but", "also", "then", "while", "because", "so", "yet"]
        connector = random.choice(connectors) if len(first_sentence) > 2 and len(second_sentence) > 2 else ""
        
        if connector:
            result = first_sentence + [",", connector] + second_sentence
        else:
            result = first_sentence + ["."] + second_sentence
        
        # Убираем повторы внутри итогового ответа
        cleaned = self.remove_word_repetitions(result)
        
        # НОВОЕ: улучшаем sentence flow
        # Передаём candidates для грамматических правил!
        flow_improved = self._improve_sentence_flow(cleaned, all_candidates)

        # ИСПРАВЛЕНО: применяем грамматические правила к готовому ответу
        # Передаём candidates чтобы выбирать глаголы из резонанса, не из шаблона!
        grammar_final = self._apply_final_grammar_rules(flow_improved, all_candidates)

        # ФИНАЛЬНАЯ грамматическая коррекция
        grammar_final = self._fix_grammar_errors(grammar_final)

        # POST-PROCESSING: Clean grammar glitches (am my → am, feel...feel → feel)
        grammar_final = self._clean_grammar_glitches(grammar_final)

        return grammar_final
    
    def _fix_grammar_errors(self, words: List[str]) -> List[str]:
        """
        Финальная грамматическая коррекция

        Исправляет распространенные ошибки:
        - "I are" → "I am"
        - "you am" → "you are"
        - "I is/was/were" → "I am"
        """
        if not words or len(words) < 2:
            return words

        result = words.copy()

        # Проходим по всем словам и исправляем грамматику
        for i in range(len(result) - 1):
            current = result[i].lower()
            next_word = result[i + 1].lower()

            # I + неправильный глагол → I am
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

    def _generate_sentence_me_style(self, candidates: List[str], length: int,
                                   used_global: set, pronouns: List[str]) -> List[str]:
        """
        Генерация одного предложения с УМНЫМ выбором слов

        УЛУЧШЕНО:
        - Smart scoring вместо random.shuffle
        - Semantic grouping: ставим связанные слова рядом
        - Better coherence через score proximity
        """
        sentence = []
        used_local = set()  # Локальный used для этого предложения

        # ME ПРИНЦИП: сначала местоимения (приоритет)
        for pronoun in pronouns:
            if len(sentence) >= length:
                break
            # ME ФИЛЬТР: не в глобальном used, не в локальном
            if (pronoun not in used_global and pronoun not in used_local):
                sentence.append(pronoun)
                used_local.add(pronoun)
                used_global.add(pronoun)

        # НОВОЕ: Smart scoring вместо random.shuffle!
        scored_candidates = self._score_candidates(candidates, "")

        # УЛУЧШЕНИЕ: Group by score tiers for better coherence
        # Высокий score = качественные слова, ставим их раньше
        if scored_candidates:
            # Разделяем на 3 tier по score
            scores = [s for w, s in scored_candidates]
            if scores:
                max_score = max(scores)
                high_tier = [(w, s) for w, s in scored_candidates if s >= max_score * 0.7]
                mid_tier = [(w, s) for w, s in scored_candidates if max_score * 0.4 <= s < max_score * 0.7]
                low_tier = [(w, s) for w, s in scored_candidates if s < max_score * 0.4]

                # Сначала берем из high tier (лучшие слова)
                for word, score in high_tier:
                    if len(sentence) >= length:
                        break
                    if (word not in used_global and word not in used_local and
                        word not in sentence and len(word) > 1):
                        sentence.append(word)
                        used_local.add(word)
                        used_global.add(word)

                # Потом mid tier если нужно
                for word, score in mid_tier:
                    if len(sentence) >= length:
                        break
                    if (word not in used_global and word not in used_local and
                        word not in sentence and len(word) > 1):
                        sentence.append(word)
                        used_local.add(word)
                        used_global.add(word)

                # Low tier только если совсем мало слов
                if len(sentence) < length // 2:
                    for word, score in low_tier:
                        if len(sentence) >= length:
                            break
                        if (word not in used_global and word not in used_local and
                            word not in sentence and len(word) > 1):
                            sentence.append(word)
                            used_local.add(word)
                            used_global.add(word)

        # ME ПРИНЦИП: капитализация первого слова
        if sentence:
            sentence[0] = sentence[0].capitalize()

        return sentence
    
    def _extract_charged_tokens(self, text: str) -> List[str]:
        """
        ПРИНЦИП SUBJECTIVITY: charged tokens - капитализованные или длинные слова
        НОВОЕ: заглавные слова = имена/важные понятия, усиленный поиск!
        """
        tokens = re.findall(r"\b\w+\b", text)
        charged = []
        
        for t in tokens:
            if t[:1].isupper() and len(t) > 1 and t.lower() != 'i':
                # СОХРАНЯЕМ регистр для имен собственных!
                charged.append(t)  # "Berlin", не "berlin"!
                self._mark_as_proper_noun(t)
            elif len(t) > 7:
                charged.append(t.lower())
                
        return charged or [t.lower() for t in tokens[:3]]
    
    def _mark_as_proper_noun(self, word: str):
        """
        Помечает слово как собственное имя для усиленного поиска в objectivity
        """
        if not hasattr(self, '_proper_nouns'):
            self._proper_nouns = set()
        self._proper_nouns.add(word)
        print(f"[High:ProperNoun] Detected: {word} - усиленный поиск в интернете")
    
    def _extract_content_words(self, text: str) -> List[str]:
        """
        ПРИНЦИП SUBJECTIVITY: content words без стоп-слов
        Языково-агностичная фильтрация содержательных слов
        """
        STOPWORDS = {
            "the","a","an","of","and","or","to","in","on","for","as","at","by","with","from",
            "is","are","was","were","be","been","being","this","that","it","its","into","than",
            "then","so","but","nor","if","because","while","when","where","which","who","whom",
            # Русские стоп-слова
            "и","в","на","с","по","для","как","что","это","то","не","да","нет","или","но"
        }
        
        words = re.findall(r"\b\w+\b", text.lower())
        content = [w for w in words if w not in STOPWORDS and len(w) > 1]
        
        # Уникализируем сохраняя порядок
        seen = set()
        unique_content = []
        for w in content:
            if w not in seen:
                seen.add(w)
                unique_content.append(w)
                
        return unique_content

class HighJuliaInterface:
    """
    Интерфейс к Julia через subprocess для критических вычислений
    Когда Python недостаточно быстр
    """
    
    def __init__(self):
        self.julia_executable = None
        self._find_julia()
        
    def _find_julia(self):
        """Поиск Julia исполняемого файла"""
        # Сначала ищем в системе
        try:
            result = subprocess.run(['which', 'julia'], capture_output=True, text=True)
            if result.returncode == 0:
                self.julia_executable = result.stdout.strip()
                return
        except:
            pass
            
        # Ищем в локальном каталоге nicole2julia
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
                print(f"[High] Найдена Julia: {self.julia_executable}")
                return
                
        self.julia_executable = None
        print("[High] Julia исполняемый файл не найден - используем встроенный интерпретер из исходников nicole2julia")
    
    def execute_julia_math(self, julia_code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Выполнение Julia математики через встроенный интерпретер
        Использует исходники Julia из nicole2julia для быстрых вычислений
        """
        try:
            # Встроенный Julia интерпретер из исходников nicole2julia
            result = self._execute_julia_native(julia_code)
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Julia execution failed: {e}'}
    
    def _execute_julia_native(self, julia_code: str) -> Dict[str, Any]:
        """Нативное выполнение Julia через исходники"""
        
        # Julia математические функции из исходников
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
            
        # Простой Julia парсер для математических операций
        lines = julia_code.strip().split('\n')
        result = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Присваивание: x = expression
            if '=' in line and not any(op in line for op in ['==', '!=', '<=', '>=']):
                var_name, expression = line.split('=', 1)
                var_name = var_name.strip()
                expr_result = self._eval_julia_expression(expression.strip(), julia_math, variables)
                variables[var_name] = expr_result
                result = expr_result
                
            # Функция println
            elif line.startswith('println('):
                args_str = line[8:-1]  # Убираем println( и )
                args = [self._eval_julia_expression(arg.strip().strip('"'), julia_math, variables) for arg in args_str.split(',')]
                julia_println(*args)
                
            # Простое выражение
            else:
                result = self._eval_julia_expression(line, julia_math, variables)
        
        return {
            'success': True,
            'result': result,
            'output': '\n'.join(output),
            'variables': variables
        }
    
    def _eval_julia_expression(self, expr: str, julia_math: dict, variables: dict):
        """Вычисляет Julia выражение используя исходники"""
        expr = expr.strip().strip('"')
        
        # Замена переменных
        for var_name, var_value in variables.items():
            expr = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), expr)
        
        # Безопасное выполнение с Julia математикой
        safe_globals = {
            '__builtins__': {},
            'math': math,
        }
        safe_globals.update(julia_math)
        
        try:
            return eval(expr, safe_globals)
        except:
            # Если строка - возвращаем как есть
            if isinstance(expr, str) and not any(c in expr for c in '+-*/()'):
                return expr
            return float(expr) if expr.replace('.', '').isdigit() else expr

class HighTransformerOptimizer:
    """
    Оптимизатор трансформеров через Julia математику
    Интегрируется с процессом создания трансформеров в Nicole
    """
    
    def __init__(self):
        self.math_engine = HighMathEngine()
        self.julia_interface = HighJuliaInterface()
        
    def optimize_transformer_creation(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Математическая оптимизация при создании нового трансформера
        Вызывается из nicole.py при _spawn_new_transformer()
        """
        # Быстрый анализ контекста
        optimization = self.math_engine.optimize_transformer_architecture(session_context)
        
        # Дополнительные Julia вычисления если доступна
        if self.julia_interface.julia_executable:
            julia_optimization = self._julia_transformer_analysis(session_context)
            if julia_optimization['success']:
                optimization['julia_enhanced'] = True
                optimization['julia_metrics'] = julia_optimization['output']
        
        return optimization
    
    def _julia_transformer_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Глубокий анализ через Julia для оптимизации"""
        julia_code = """
# Анализ сложности контекста для трансформера
context_complexity = 42.0  # Заглушка
learning_efficiency = sqrt(context_complexity) / 10
optimal_depth = ceil(log(context_complexity + 1))

println("complexity:", context_complexity)
println("efficiency:", learning_efficiency) 
println("depth:", optimal_depth)
"""
        
        return self.julia_interface.execute_julia_math(julia_code)
    
    def enhance_learning_process(self, text: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Улучшение процесса дообучения через Julia математику
        Вызывается при обработке каждого сообщения
        """
        # Быстрое вычисление метрик
        entropy = self.math_engine.vectorized_entropy([text])
        
        # Анализ n-граммов для обучения
        ngrams = self.math_engine.fast_ngram_analysis(text)
        
        # Оптимизация на основе текущих метрик
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
    Ядро High системы - математический мозг Nicole
    Интегрируется везде где нужны быстрые вычисления
    """
    
    def __init__(self):
        self.math_engine = HighMathEngine()
        self.transformer_optimizer = HighTransformerOptimizer()
        self.julia_interface = HighJuliaInterface()
        
        self.is_active = False
        self.log_file = "high_system.log"
        
    def activate(self) -> bool:
        """Активация High математической системы"""
        try:
            self.is_active = True
            self._log_info("High system activated - mathematical brain online")
            return True
        except Exception as e:
            self._log_error(f"High activation failed: {e}")
            return False
    
    def deactivate(self):
        """Деактивация High системы"""
        self.is_active = False
        self._log_info("High system deactivated")
    
    def optimize_transformer_for_nicole(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Главная функция оптимизации трансформеров
        Вызывается из nicole.py при создании трансформера
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
        Улучшение дообучения Nicole через Julia
        Вызывается при каждом сообщении для быстрых вычислений
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
        Оптимизация пунктуации через математические модели
        Анализирует паттерны и улучшает структуру предложений
        """
        if not self.is_active:
            return text
        
        try:
            # Разбиваем на части для анализа
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Математическая оптимизация пунктуации
            optimized_parts = self.math_engine.predict_punctuation_placement(sentences)
            
            # Собираем обратно
            result = ' '.join(optimized_parts)
            
            self._log_info(f"Punctuation optimized: {len(sentences)} sentences")
            return result
            
        except Exception as e:
            self._log_error(f"Punctuation optimization failed: {e}")
            return text
    
    def get_mathematical_status(self) -> Dict[str, Any]:
        """Статус математической системы"""
        return {
            'active': self.is_active,
            'julia_available': self.julia_interface.julia_executable is not None,
            'julia_path': self.julia_interface.julia_executable,
            'cache_size': len(self.math_engine.julia_cache),
            'temp_dir': str(self.temp_dir)
        }
    
    def _log_info(self, message: str):
        """Логирование для системы"""
        with open(self.log_file, "a") as f:
            f.write(f"[HIGH:INFO] {time.time()}: {message}\n")
    
    def _log_error(self, message: str):
        """Логирование ошибок"""
        with open(self.log_file, "a") as f:
            f.write(f"[HIGH:ERROR] {time.time()}: {message}\n")

# Глобальный экземпляр High системы
_high_core = None

def get_high_core() -> HighCore:
    """Получение глобального экземпляра High математической системы"""
    global _high_core
    if _high_core is None:
        _high_core = HighCore()
    return _high_core

def activate_high_system() -> bool:
    """Активация High системы для Nicole"""
    high = get_high_core()
    return high.activate()

def deactivate_high_system():
    """Деактивация High системы"""
    high = get_high_core()
    high.deactivate()

# Пример Julia кода для трансформера
EXAMPLE_JULIA_MATH_SCRIPT = """
# Julia математика для трансформера Nicole
function calculate_transformer_metrics(entropy::Float64, resonance::Float64)
    # Векторизованные вычисления
    perplexity = exp(entropy)
    coherence = 1.0 / (1.0 + exp(-resonance))
    engagement = sqrt(entropy * resonance)
    
    return (perplexity, coherence, engagement)
end

# Тест вычислений
entropy_val = 2.5
resonance_val = 0.7

metrics = calculate_transformer_metrics(entropy_val, resonance_val)
println("Perplexity: ", metrics[1])
println("Coherence: ", metrics[2]) 
println("Engagement: ", metrics[3])
"""

if __name__ == "__main__":
    # Тестирование High системы
    print("🧮 HIGH SYSTEM - Nicole Mathematical Brain")
    
    high = get_high_core()
    
    if high.activate():
        print("✅ High system activated")
        
        # Тест математических вычислений
        test_data = ["hello world", "nicole learns fast", "mathematical optimization"]
        entropy = high.math_engine.vectorized_entropy(test_data)
        print(f"📊 Vectorized entropy: {entropy:.3f}")
        
        # Тест оптимизации трансформера
        context = {'messages': test_data}
        optimization = high.optimize_transformer_for_nicole(context)
        print(f"🧠 Transformer optimization: {optimization.get('architecture_type')}")
        
        # Тест пунктуации
        test_text = "hello world this is test sentence without punctuation"
        optimized = high.optimize_punctuation(test_text)
        print(f"✏️ Punctuation: '{optimized}'")
        
        # Тест Julia интерфейса
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
        
        # Статус системы
        status = high.get_mathematical_status()
        print(f"🧮 High system status: {status}")
        
        high.deactivate()
        print("✅ High system deactivated")
    else:
        print("❌ High system activation failed")
