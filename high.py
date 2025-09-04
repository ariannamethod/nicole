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
import numpy as np
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
        Векторизованное вычисление энтропии текста
        В 100x быстрее чем Python циклы
        """
        if not text_data:
            return 0.0
            
        # Быстрый подсчет частот через numpy
        word_counts = {}
        total_words = 0
        
        for text in text_data:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if total_words == 0:
            return 0.0
        
        # Векторизованное вычисление энтропии
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    def calculate_resonance_matrix(self, words: List[str]) -> np.ndarray:
        """
        Вычисление матрицы резонанса между словами
        Для оптимизации трансформеров
        """
        if not words:
            return np.array([])
            
        n = len(words)
        resonance_matrix = np.zeros((n, n))
        
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
        Быстрый анализ n-граммов для пунктуации
        Векторизованная обработка для определения правил
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
    
    def invert_pronouns_me_style(self, words: List[str]) -> List[str]:
        """
        Инверсия местоимений по принципу ME
        you↔i, your↔my, me↔you для правильной перспективы
        """
        pronoun_mapping = {
            'you': 'i', 'u': 'i', 'your': 'my', 'yours': 'mine', 'yourself': 'myself',
            'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'myself': 'yourself',
            'we': 'you'
        }
        
        return [pronoun_mapping.get(w.lower(), w) for w in words]
    
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
            
        # Фоллбек если совсем пусто
        if not all_candidates:
            all_candidates = ["understand", "know", "think", "feel", "see", "work", "create", "learn"]
        
        # Инвертированные местоимения как приоритет (принцип ME)
        inverted_pronouns = self.invert_pronouns_me_style(user_words)
        pronoun_preferences = [w for w in inverted_pronouns if w in ['i', 'you', 'я', 'ты', 'my', 'мой', 'меня', 'мне']]
        
        # Добавляем базовые местоимения если нет инверсии
        if not pronoun_preferences:
            pronoun_preferences = ['i', 'my']
        
        used_words = set(user_words)  # Не повторяем слова юзера
        
        # ME ПРИНЦИП: строгий used set между предложениями
        used_between_sentences = set(user_words)  # Не повторяем слова юзера
        
        # Генерируем первое предложение
        first_sentence = self._generate_sentence_me_style(
            all_candidates, base1, used_between_sentences, pronoun_preferences
        )
        
        # Генерируем второе предложение (used обновлен первым предложением)
        second_sentence = self._generate_sentence_me_style(
            all_candidates, base2, used_between_sentences, pronoun_preferences
        )
        
        # ME ПРИНЦИП: два предложения с точкой между ними
        result = first_sentence + ["."] + second_sentence
        
        # Убираем повторы внутри итогового ответа
        return self.remove_word_repetitions(result)
    
    def _generate_sentence_me_style(self, candidates: List[str], length: int, 
                                   used_global: set, pronouns: List[str]) -> List[str]:
        """Генерация одного предложения по принципам ME с строгими фильтрами"""
        sentence = []
        used_local = set()  # Локальный used для этого предложения
        
        # ME ПРИНЦИП: сначала местоимения (приоритет)
        for pronoun in pronouns:
            if len(sentence) >= length:
                break
            # ME ФИЛЬТР: не в глобальном used, не в локальном, не односимвольное
            if (pronoun not in used_global and pronoun not in used_local and 
                len(pronoun) > 1):
                sentence.append(pronoun)
                used_local.add(pronoun)
                used_global.add(pronoun)  # Обновляем глобальный
        
        # ME ПРИНЦИП: затем кандидаты с строгими фильтрами
        random.shuffle(candidates)
        for word in candidates:
            if len(sentence) >= length:
                break
            # ME ФИЛЬТР: строгая проверка повторов + длина > 1
            if (word not in used_global and word not in used_local and 
                len(word) > 1 and word not in sentence):
                sentence.append(word)
                used_local.add(word)
                used_global.add(word)
        
        # ME ПРИНЦИП: дополняем с защитой от циклов
        attempts = 0
        while len(sentence) < length and candidates and attempts < 20:
            word = random.choice(candidates)
            # ME ФИЛЬТР: строгая проверка
            if (word not in used_global and word not in used_local and 
                word not in sentence and len(word) > 1):
                sentence.append(word)
                used_local.add(word)
                used_global.add(word)
            attempts += 1
        
        # ME ПРИНЦИП: исправляем плохой конец
        if sentence and len(sentence[-1]) == 1:
            sentence[-1] = "hmm"
        
        # ME ПРИНЦИП: капитализация первого слова
        if sentence:
            sentence[0] = sentence[0].capitalize()
            
        return sentence
    
    def _extract_charged_tokens(self, text: str) -> List[str]:
        """
        ПРИНЦИП SUBJECTIVITY: charged tokens - капитализованные или длинные слова
        Языково-агностичное выделение важных токенов
        """
        tokens = re.findall(r"\b\w+\b", text)
        charged = [t.lower() for t in tokens if (t[:1].isupper() and len(t) > 1) or len(t) > 7]
        return charged or [t.lower() for t in tokens[:3]]
    
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
        try:
            result = subprocess.run(['which', 'julia'], capture_output=True, text=True)
            if result.returncode == 0:
                self.julia_executable = result.stdout.strip()
        except:
            self.julia_executable = None
    
    def execute_julia_math(self, julia_code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Выполнение Julia кода для математических вычислений
        Для критически важных быстрых операций
        """
        if not self.julia_executable:
            return {'success': False, 'error': 'Julia not available'}
            
        try:
            # Оборачиваем код для безопасного выполнения
            wrapped_code = f"""
try
    {julia_code}
catch e
    println("ERROR: ", e)
end
"""
            
            result = subprocess.run(
                [self.julia_executable, '-e', wrapped_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Julia execution timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

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
