#!/usr/bin/env python3
"""
NICOLE - Neural Intelligent Conversational Organism Language Engine
Флюидная нейронка без весов, создающая уникальные трансформеры для каждого диалога.
Посвящается Лео.
"""

import sys
import os
# Добавляем текущую директорию в путь для импорта наших модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h2o
import sqlite3
import json
import time
import random
import math
import threading
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Импорт ME принципов из nicole_metrics
try:
    from nicole_metrics import MEPunctuationFilters, VerbGraph, ResonanceAnalyzer
except ImportError:
    # Заглушки если модуль недоступен
    class MEPunctuationFilters:
        @staticmethod
        def apply_all_filters(text): return text
    class VerbGraph:
        def __init__(self): pass
        def analyze_text_for_verbs(self, text): pass
        def predict_verb_ending(self, verb): return "."
    class ResonanceAnalyzer:
        @staticmethod
        def find_resonant_word(text, freq=None): return "", 0.0

# Импорт Objectivity для динамических весов
try:
    from nicole_objectivity import nicole_objectivity
except ImportError:
    # Заглушка если модуль недоступен
    class MockObjectivity:
        async def create_dynamic_context(self, msg, metrics): return []
        def extract_response_seeds(self, context, percent=0.5): return []
        def format_context_for_nicole(self, windows): return ""
    nicole_objectivity = MockObjectivity()

@dataclass
class ConversationMetrics:
    """Метрики текущего разговора"""
    entropy: float = 0.0
    perplexity: float = 0.0
    resonance: float = 0.0
    coherence: float = 0.0
    engagement: float = 0.0
    
class NicoleMemory:
    """Система памяти Nicole без весов + принципы ME"""
    
    def __init__(self, db_path: str = "nicole_memory.db"):
        self.db_path = db_path
        self.word_frequencies = defaultdict(int)
        self.bigram_transitions = defaultdict(lambda: defaultdict(int))
        self.verb_graph = VerbGraph()
        self.init_database()
        self.load_persistent_memory()  # Загружаем существующую память
        
    def init_database(self):
        """Инициализация базы данных памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # ME принципы: таблица биграмм
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bigrams (
            id INTEGER PRIMARY KEY,
            w1 TEXT,
            w2 TEXT,
            count INTEGER DEFAULT 1,
            UNIQUE(w1, w2)
        )
        """)
        
        # ME принципы: частоты слов
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT PRIMARY KEY,
            count INTEGER DEFAULT 1
        )
        """)
        
        # Таблица для отслеживания первых контактов с юзерами
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_first_contact (
            user_id TEXT PRIMARY KEY,
            first_contact_time REAL,
            template_phase_completed INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0
        )
        """)
        
        # Добавляем колонку message_count если ее нет (для существующих баз)
        try:
            cursor.execute("ALTER TABLE user_first_contact ADD COLUMN message_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Колонка уже существует
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transformer_logs (
            id INTEGER PRIMARY KEY,
            transformer_id TEXT,
            session_id TEXT,
            creation_time REAL,
            death_time REAL,
            architecture TEXT,
            performance_metrics TEXT,
            evolution_history TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        
    def log_conversation(self, session_id: str, user_input: str, nicole_output: str, 
                        metrics: ConversationMetrics, transformer_config: Dict):
        """Логирует разговор"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO conversations 
        (session_id, timestamp, user_input, nicole_output, metrics, transformer_config)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            time.time(),
            user_input,
            nicole_output,
            json.dumps(metrics.__dict__),
            json.dumps(transformer_config)
        ))
        
        conn.commit()
        conn.close()
        
    def log_transformer_lifecycle(self, transformer_id: str, session_id: str,
                                 architecture: Dict, creation_time: float,
                                 death_time: float = None, performance: Dict = None):
        """Логирует жизненный цикл трансформера"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO transformer_logs 
        (transformer_id, session_id, creation_time, death_time, architecture, performance_metrics)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            transformer_id,
            session_id,
            creation_time,
            death_time,
            json.dumps(architecture),
            json.dumps(performance or {})
        ))
        
        conn.commit()
        conn.close()
        
    def update_word_frequencies(self, text: str):
        """Обновляет частоты слов из ME принципов"""
        words = text.lower().split()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for word in words:
            self.word_frequencies[word] += 1
            cursor.execute("""
            INSERT OR REPLACE INTO word_frequencies (word, count)
            VALUES (?, COALESCE((SELECT count FROM word_frequencies WHERE word = ?), 0) + 1)
            """, (word, word))
            
        conn.commit()
        conn.close()
        
    def update_bigrams(self, text: str):
        """Обновляет биграммы из ME для марковских цепей"""
        words = text.lower().split()
        if len(words) < 2:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigram_transitions[w1][w2] += 1
            
            cursor.execute("""
            INSERT OR REPLACE INTO bigrams (w1, w2, count)
            VALUES (?, ?, COALESCE((SELECT count FROM bigrams WHERE w1 = ? AND w2 = ?), 0) + 1)
            """, (w1, w2, w1, w2))
            
        conn.commit()
        conn.close()
        
    def get_semantic_candidates(self, resonant_word: str, distance_percent: float = 0.5) -> List[str]:
        """Получает кандидатов на семантической дистанции от резонантного слова (ME принцип)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получаем все слова из истории
        cursor.execute("SELECT word, count FROM word_frequencies ORDER BY count DESC LIMIT 200")
        word_data = cursor.fetchall()
        conn.close()
        
        if not word_data:
            return [resonant_word]
            
        candidates = []
        target_distance = distance_percent
        
        for word, freq in word_data:
            if word == resonant_word:
                continue
                
            # Простая семантическая дистанция через частоты
            resonant_freq = self.word_frequencies.get(resonant_word, 1)
            word_freq = freq
            
            # Нормализуем частоты и считаем дистанцию
            max_freq = max(resonant_freq, word_freq)
            min_freq = min(resonant_freq, word_freq)
            distance = 1.0 - (min_freq / max_freq) if max_freq > 0 else 1.0
            
            # Берем слова близкие к целевой дистанции
            if abs(distance - target_distance) < 0.2:
                candidates.append(word)
                
        return candidates[:10]  # Ограничиваем количество
        
    def load_persistent_memory(self):
        """Загружает существующую память из SQLite при старте"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Загружаем частоты слов
            cursor.execute("SELECT word, count FROM word_frequencies")
            for word, count in cursor.fetchall():
                self.word_frequencies[word] = count
                
            # Загружаем биграммы
            cursor.execute("SELECT w1, w2, count FROM bigrams")
            for w1, w2, count in cursor.fetchall():
                self.bigram_transitions[w1][w2] = count
                
            conn.close()
            
            total_words = len(self.word_frequencies)
            total_bigrams = sum(len(transitions) for transitions in self.bigram_transitions.values())
            print(f"[Nicole:Memory] Загружена память: {total_words} слов, {total_bigrams} биграмм")
            
        except Exception as e:
            print(f"[Nicole:Memory] Ошибка загрузки памяти: {e}")
    
    def is_response_repetitive(self, response: str, user_id: str = None, limit: int = 5) -> bool:
        """Проверяет не повторяется ли ответ (анти-повтор логика)"""
        if not user_id:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ищем последние ответы этого юзера
            cursor.execute("""
            SELECT nicole_output FROM conversations 
            WHERE session_id LIKE ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """, (f"%{user_id}%", limit))
            
            recent_responses = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Проверяем точное совпадение
            if response in recent_responses:
                return True
                
            # Проверяем похожесть (больше 80% общих слов)
            response_words = set(response.lower().split())
            for past_response in recent_responses:
                past_words = set(past_response.lower().split())
                if len(response_words) > 0 and len(past_words) > 0:
                    similarity = len(response_words & past_words) / len(response_words | past_words)
                    if similarity > 0.8:
                        return True
                        
            return False
            
        except Exception as e:
            print(f"[Nicole:Memory] Ошибка проверки повторов: {e}")
            return False

class FluidTransformer:
    """Флюидный трансформер без предобученных весов"""
    
    def __init__(self, transformer_id: str, session_context: Dict = None):
        self.transformer_id = transformer_id
        self.session_context = session_context or {}
        self.architecture = self._generate_initial_architecture()
        self.creation_time = time.time()
        self.last_evolution = time.time()
        self.conversation_history = []
        self.current_metrics = ConversationMetrics()
        
    def _generate_initial_architecture(self) -> Dict:
        """Генерирует случайную начальную архитектуру"""
        return {
            'attention_heads': random.randint(2, 8),
            'hidden_dim': random.choice([64, 128, 256, 512]),
            'num_layers': random.randint(2, 6),
            'vocab_size': 1000,  # Начальный размер словаря
            'context_window': random.randint(128, 1024),
            'dropout_rate': random.uniform(0.1, 0.3),
            'learning_rate': random.uniform(0.0001, 0.01),
            'temperature': random.uniform(0.5, 1.5),
            'top_k': random.randint(5, 50),
            'top_p': random.uniform(0.7, 0.95),
        }
        
    def generate_transformer_script(self) -> str:
        """Генерирует код трансформера на основе архитектуры"""
        arch = self.architecture
        
        script = f"""
# Флюидный трансформер {self.transformer_id}
# Архитектура: {arch['num_layers']} слоев, {arch['attention_heads']} голов внимания

import math
import random

class AttentionHead:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
    def forward(self, query, key, value):
        # Упрощенное внимание без весов
        attention_scores = []
        for i in range(len(query)):
            score = sum(q * k for q, k in zip(query[i], key[i])) * self.scale
            attention_scores.append(math.tanh(score))
        
        # Софтмакс
        exp_scores = [math.exp(score) for score in attention_scores]
        sum_exp = sum(exp_scores)
        attention_weights = [score / sum_exp for score in exp_scores]
        
        # Взвешенная сумма значений
        output = []
        for i in range(len(value[0])):
            weighted_sum = sum(w * value[j][i] for j, w in enumerate(attention_weights))
            output.append(weighted_sum)
            
        return output, attention_weights

class FluidLayer:
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_heads = [AttentionHead(hidden_dim // num_heads) for _ in range(num_heads)]
        
    def forward(self, x):
        # Мульти-головое внимание
        head_outputs = []
        attention_maps = []
        
        for head in self.attention_heads:
            head_out, attn_weights = head.forward(x, x, x)  # Self-attention
            head_outputs.append(head_out)
            attention_maps.append(attn_weights)
            
        # Конкатенация голов
        combined = []
        for i in range(len(head_outputs[0])):
            combined.append(sum(head[i] for head in head_outputs) / len(head_outputs))
            
        # Residual connection + простая нормализация
        output = []
        for i in range(len(x[0])):
            residual = x[0][i] + combined[i]
            # Простая нормализация
            output.append(math.tanh(residual))
            
        return [output], attention_maps

class H2OTransformer:
    def __init__(self):
        self.num_layers = {arch['num_layers']}
        self.hidden_dim = {arch['hidden_dim']}
        self.num_heads = {arch['attention_heads']}
        self.context_window = {arch['context_window']}
        self.temperature = {arch['temperature']}
        
        self.layers = [FluidLayer(self.hidden_dim, self.num_heads) for _ in range(self.num_layers)]
        self.vocab_embedding = self._init_embedding({arch['vocab_size']}, self.hidden_dim)
        
        h2o_log(f"Трансформер инициализирован: {{self.num_layers}} слоев, {{self.num_heads}} голов")
        
    def _init_embedding(self, vocab_size, hidden_dim):
        # Случайная инициализация эмбеддингов
        embedding = []
        for i in range(vocab_size):
            vector = [random.gauss(0, 0.1) for _ in range(hidden_dim)]
            embedding.append(vector)
        return embedding
        
    def tokenize(self, text):
        # Простая токенизация по словам
        words = text.lower().split()
        tokens = []
        for word in words:
            token_id = hash(word) % len(self.vocab_embedding)
            tokens.append(token_id)
        return tokens
        
    def embed_tokens(self, tokens):
        # Получаем эмбеддинги для токенов
        embeddings = []
        for token in tokens:
            if token < len(self.vocab_embedding):
                embeddings.append(self.vocab_embedding[token])
            else:
                # Случайный эмбеддинг для неизвестных токенов
                embeddings.append([random.gauss(0, 0.1) for _ in range(self.hidden_dim)])
        return embeddings
        
    def forward(self, input_text):
        h2o_log(f"Обрабатываем: '{{input_text}}'")
        
        # Токенизация и эмбеддинг
        tokens = self.tokenize(input_text)
        embeddings = self.embed_tokens(tokens)
        
        if not embeddings:
            return "..."
            
        # Проходим через слои
        x = embeddings
        all_attention_maps = []
        
        for i, layer in enumerate(self.layers):
            x, attention_maps = layer.forward(x)
            all_attention_maps.append(attention_maps)
            h2o_metric(f"layer_{{i}}_output_norm", sum(sum(abs(val) for val in row) for row in x))
            
        # Простая генерация ответа
        output_logits = x[0]  # Берем первый элемент последовательности
        
        # Применяем температуру
        scaled_logits = [logit / self.temperature for logit in output_logits]
        
        # Простой сэмплинг
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
        sum_exp = sum(exp_logits)
        probs = [exp_logit / sum_exp for exp_logit in exp_logits]
        
        # Выбираем слова на основе вероятностей
        response_words = []
        for _ in range(min(10, len(tokens) + 2)):  # Ограничиваем длину ответа
            # Простой сэмплинг
            r = random.random()
            cumsum = 0
            selected_idx = 0
            for i, prob in enumerate(probs):
                cumsum += prob
                if r <= cumsum:
                    selected_idx = i
                    break
                    
            # Генерируем слово (упрощенно)
            word_candidates = ["да", "нет", "может", "интересно", "понятно", "хорошо", "плохо", "как", "что", "зачем"]
            if selected_idx < len(word_candidates):
                response_words.append(word_candidates[selected_idx])
            else:
                response_words.append("...")
                
        response = " ".join(response_words)
        h2o_log(f"Сгенерирован ответ: '{{response}}'")
        
        return response
        
    def calculate_metrics(self, input_text, output_text):
        # Рассчитываем метрики для эволюции
        entropy = len(set(input_text.split())) / max(1, len(input_text.split()))
        perplexity = len(output_text.split()) / max(1, len(input_text.split()))
        resonance = len(set(input_text.split()) & set(output_text.split())) / max(1, len(set(input_text.split())))
        
        h2o_metric("entropy", entropy)
        h2o_metric("perplexity", perplexity) 
        h2o_metric("resonance", resonance)
        
        return entropy, perplexity, resonance

# Глобальная переменная для трансформера
transformer = None

def init_transformer():
    global transformer
    if transformer is None:
        transformer = H2OTransformer()
    return transformer

def process_input(user_input):
    t = init_transformer()
    response = t.forward(user_input)
    metrics = t.calculate_metrics(user_input, response)
    h2o_log(f"Метрики: entropy={{metrics[0]:.3f}}, perplexity={{metrics[1]:.3f}}, resonance={{metrics[2]:.3f}}")
    return response

h2o_log("=== H2O ТРАНСФОРМЕР ГОТОВ ===")
"""
        
        return script
        
    def evolve_architecture(self, metrics: ConversationMetrics):
        """Эволюционирует архитектуру на основе метрик"""
        old_arch = self.architecture.copy()
        
        # Адаптивные изменения на основе метрик
        if metrics.entropy < 0.3:  # Низкое разнообразие
            self.architecture['num_heads'] = min(16, self.architecture['num_heads'] + 1)
            self.architecture['hidden_dim'] = min(1024, int(self.architecture['hidden_dim'] * 1.2))
            
        if metrics.perplexity > 2.0:  # Высокая сложность
            self.architecture['num_layers'] = min(12, self.architecture['num_layers'] + 1)
            self.architecture['context_window'] = min(2048, int(self.architecture['context_window'] * 1.5))
            
        if metrics.resonance < 0.2:  # Плохой резонанс
            self.architecture['temperature'] = max(0.1, self.architecture['temperature'] * 0.8)
            self.architecture['top_p'] = max(0.5, self.architecture['top_p'] * 0.9)
            
        if metrics.coherence < 0.4:  # Низкая связность
            self.architecture['dropout_rate'] = max(0.05, self.architecture['dropout_rate'] * 0.8)
            
        # Логируем эволюцию
        changes = {}
        for key, value in self.architecture.items():
            if old_arch[key] != value:
                changes[key] = {'old': old_arch[key], 'new': value}
                
        if changes:
            print(f"[Nicole] Трансформер {self.transformer_id} эволюционировал: {changes}")
            self.last_evolution = time.time()
            
        return len(changes) > 0
        
    def should_die(self) -> bool:
        """Определяет, должен ли трансформер умереть"""
        # Умирает если:
        # 1. Прошло много времени без эволюции
        # 2. Плохие метрики долгое время
        # 3. Случайная смерть для обновления
        
        time_since_creation = time.time() - self.creation_time
        time_since_evolution = time.time() - self.last_evolution
        
        if time_since_creation > 300:  # 5 минут максимум
            return True
            
        if time_since_evolution > 120:  # 2 минуты без эволюции
            return True
            
        if random.random() < 0.01:  # 1% случайная смерть
            return True
            
        return False

class NicoleCore:
    """Ядро системы Nicole"""
    
    def __init__(self):
        self.memory = NicoleMemory()
        self.h2o_engine = h2o.h2o_engine
        self.current_transformer = None
        self.session_id = None
        self.conversation_count = 0
        self.lock = threading.Lock()
        
    def start_conversation(self, session_id: str = None):
        """Начинает новый разговор"""
        if not session_id:
            session_id = f"nicole_{int(time.time() * 1000)}"
            
        self.session_id = session_id
        self.conversation_count = 0
        
        # Запускаем H2O сессию
        self.h2o_engine.start_session(session_id)
        
        # Создаем первый трансформер
        self._spawn_new_transformer()
        
        print(f"[Nicole] Начинаем разговор в сессии {session_id}")
        return session_id
        
    def _spawn_new_transformer(self):
        """Создает новый флюидный трансформер"""
        transformer_id = f"fluid_{self.session_id}_{int(time.time() * 1000000)}"
        
        # Убиваем старый трансформер если есть
        if self.current_transformer:
            self._kill_current_transformer()
            
        # Создаем новый
        self.current_transformer = FluidTransformer(transformer_id, {'session_id': self.session_id})
        
        # Генерируем и запускаем скрипт трансформера
        transformer_script = self.current_transformer.generate_transformer_script()
        
        try:
            self.h2o_engine.run_transformer_script(
                transformer_script, 
                transformer_id,
                {'session_context': self.current_transformer.session_context}
            )
            
            # Логируем создание
            self.memory.log_transformer_lifecycle(
                transformer_id,
                self.session_id,
                self.current_transformer.architecture,
                self.current_transformer.creation_time
            )
            
            print(f"[Nicole] Новый трансформер {transformer_id} создан")
            
        except Exception as e:
            print(f"[Nicole:ERROR] Ошибка создания трансформера: {e}")
            
    def _kill_current_transformer(self):
        """Убивает текущий трансформер"""
        if self.current_transformer:
            transformer_id = self.current_transformer.transformer_id
            
            # Логируем смерть
            self.memory.log_transformer_lifecycle(
                transformer_id,
                self.session_id,
                self.current_transformer.architecture,
                self.current_transformer.creation_time,
                death_time=time.time()
            )
            
            # Убиваем в H2O
            self.h2o_engine.executor.kill_transformer(transformer_id)
            
            print(f"[Nicole] Трансформер {transformer_id} уничтожен")
            self.current_transformer = None
            
    def process_message(self, user_input: str) -> str:
        """Обрабатывает сообщение пользователя с ME принципами"""
        with self.lock:
            if not self.current_transformer:
                self._spawn_new_transformer()
                
            # ME принципы: обновляем частоты слов и биграммы
            self.memory.update_word_frequencies(user_input)
            self.memory.update_bigrams(user_input)
            
            # ME принципы: находим резонантное слово
            resonant_word, resonance_score = ResonanceAnalyzer.find_resonant_word(
                user_input, self.memory.word_frequencies
            )
            
            print(f"[Nicole:ME] Резонантное слово: '{resonant_word}' (скор: {resonance_score:.3f})")
            
            # ME принципы: генерируем ответ на основе резонантного слова
            response = self._generate_me_enhanced_response(user_input, resonant_word)
            
            # ME принципы: применяем пунктуационные фильтры
            response = MEPunctuationFilters.apply_all_filters(response)
            
            # ME принципы: анализируем глаголы для будущих ответов
            self.memory.verb_graph.analyze_text_for_verbs(user_input)
            self.memory.verb_graph.analyze_text_for_verbs(response)
            
            # Обновляем метрики
            self._update_metrics(user_input, response)
            
            # Проверяем нужна ли эволюция или смерть
            self._check_transformer_lifecycle()
            
            # Логируем разговор
            self.memory.log_conversation(
                self.session_id,
                user_input,
                response,
                self.current_transformer.current_metrics,
                self.current_transformer.architecture
            )
            
            # Обновляем счетчик сообщений в SQLite (чтобы шаблоны не повторялись)
            self._update_user_message_count()
            self.conversation_count += 1
            return response
    
    async def _get_objectivity_context(self, user_input: str) -> Tuple[str, List[str]]:
        """Получает объективный контекст через динамические веса"""
        try:
            # Получаем текущие метрики
            metrics = {}
            if self.current_transformer and self.current_transformer.current_metrics:
                m = self.current_transformer.current_metrics
                metrics = {
                    'perplexity': m.perplexity,
                    'entropy': m.entropy, 
                    'resonance': m.resonance
                }
            
            # Создаем динамический контекст
            context_windows = await nicole_objectivity.create_dynamic_context(user_input, metrics)
            
            # Форматируем для Nicole
            context = nicole_objectivity.format_context_for_nicole(context_windows)
            
            # Извлекаем семена для ответа (50% из контекста)
            response_seeds = nicole_objectivity.extract_response_seeds(context, 0.5)
            
            print(f"[Nicole:Objectivity] Контекст: {len(context)} символов, семена: {response_seeds}")
            return context, response_seeds
            
        except Exception as e:
            print(f"[Nicole:Objectivity:ERROR] {e}")
            return "", []
    
    def _generate_me_enhanced_response(self, user_input: str, resonant_word: str) -> str:
        """Генерирует ответ на основе принципов ME + Objectivity"""
        try:
            # Получаем объективный контекст асинхронно
            import asyncio
            try:
                context, objectivity_seeds = asyncio.run(self._get_objectivity_context(user_input))
            except:
                context, objectivity_seeds = "", []
            
            # Получаем кандидатов на 50% и 70% семантической дистанции (как в ME)
            candidates_50 = self.memory.get_semantic_candidates(resonant_word, 0.5)
            candidates_70 = self.memory.get_semantic_candidates(resonant_word, 0.7)
            
            # Комбинируем ME кандидатов с Objectivity семенами
            all_candidates = list(set(candidates_50 + candidates_70 + objectivity_seeds))
            
            # Фильтруем только английские слова (анти-микс логика)
            english_candidates = []
            for word in all_candidates:
                # Простая проверка - если слово содержит только латинские буквы
                if word and all(ord(c) < 128 for c in word.lower()):
                    english_candidates.append(word)
            
            all_candidates = english_candidates if english_candidates else ["understand", "resonate", "learn"]
            
            if not all_candidates:
                # Фоллбек на простые ответы
                return self._generate_simple_response(user_input)
            
            # Строим ответ на основе биграмм (марковские цепи из ME)
            response_words = []
            
            # Начинаем с резонантного слова или близкого к нему
            if resonant_word and resonant_word in self.memory.bigram_transitions:
                current_word = resonant_word
            elif all_candidates:
                current_word = random.choice(all_candidates)
            else:
                current_word = "понимаю"
                
            response_words.append(current_word)
            
            # Генерируем следующие слова через биграммы (как в ME)
            for _ in range(random.randint(3, 8)):  # Длина ответа 4-9 слов
                if current_word in self.memory.bigram_transitions:
                    next_words = self.memory.bigram_transitions[current_word]
                    if next_words:
                        # Выбираем следующее слово по весам
                        total_count = sum(next_words.values())
                        r = random.randint(1, total_count)
                        cumsum = 0
                        for word, count in next_words.items():
                            cumsum += count
                            if cumsum >= r:
                                current_word = word
                                break
                    else:
                        # Если нет биграммы, берем случайный кандидат
                        current_word = random.choice(all_candidates) if all_candidates else "да"
                else:
                    current_word = random.choice(all_candidates) if all_candidates else "хорошо"
                    
                response_words.append(current_word)
            
            # Собираем ответ
            response = " ".join(response_words)
            
            # Добавляем пунктуацию на основе verb graph
            if response_words:
                last_word = response_words[-1]
                if last_word in self.memory.verb_graph.common_verbs:
                    punct = self.memory.verb_graph.predict_verb_ending(last_word)
                    if not response.endswith(('.', '!', '?')):
                        response += punct
                elif not response.endswith(('.', '!', '?')):
                    response += "."
            
            print(f"[Nicole:ME] Генерация: '{resonant_word}' -> {len(all_candidates)} кандидатов -> '{response}'")
            
            # Проверяем на повторы
            user_id = self.session_id.replace("tg_", "") if self.session_id else "unknown"
            if self.memory.is_response_repetitive(response, user_id):
                print(f"[Nicole:AntiRepeat] Ответ повторяется, генерируем альтернативу")
                return self._generate_simple_response(user_input)
            
            return response
            
        except Exception as e:
            print(f"[Nicole:ME:ERROR] Ошибка ME генерации: {e}")
            return self._generate_simple_response(user_input)
                
    def _is_first_time_user(self, user_id: str = None) -> bool:
        """Проверяет первый ли раз видим этого юзера"""
        if not user_id:
            user_id = self.session_id.replace("tg_", "") if self.session_id else "unknown"
            
        try:
            conn = sqlite3.connect(self.memory.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT template_phase_completed, message_count FROM user_first_contact WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result is None:
                # Первый раз видим этого юзера - записываем
                cursor.execute("""
                INSERT INTO user_first_contact (user_id, first_contact_time, template_phase_completed, message_count)
                VALUES (?, ?, 0, 0)
                """, (user_id, time.time()))
                conn.commit()
                conn.close()
                return True
            else:
                # Загружаем счетчик сообщений в текущую сессию
                self.conversation_count = result[1] if result[1] else 0
                conn.close()
                return result[0] == 0  # Если template_phase_completed = 0, то еще в шаблонной фазе
                
        except Exception as e:
            print(f"[Nicole] Ошибка проверки первого контакта: {e}")
            return False
    
    def _mark_template_phase_completed(self, user_id: str = None):
        """Отмечает что шаблонная фаза завершена для юзера"""
        if not user_id:
            user_id = self.session_id.replace("tg_", "") if self.session_id else "unknown"
            
        try:
            conn = sqlite3.connect(self.memory.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE user_first_contact SET template_phase_completed = 1 WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            print(f"[Nicole] Шаблонная фаза завершена для {user_id}")
        except Exception as e:
            print(f"[Nicole] Ошибка завершения шаблонной фазы: {e}")
    
    def _update_user_message_count(self, user_id: str = None):
        """Обновляет счетчик сообщений пользователя в SQLite"""
        if not user_id:
            user_id = self.session_id.replace("tg_", "") if self.session_id else "unknown"
            
        try:
            conn = sqlite3.connect(self.memory.db_path)
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE user_first_contact 
            SET message_count = message_count + 1 
            WHERE user_id = ?
            """, (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Nicole] Ошибка обновления счетчика: {e}")
    
    def _generate_simple_response(self, user_input: str) -> str:
        """Генерирует простой ответ для начального обучения"""
        # Проверяем первый ли раз видим этого юзера
        is_first_time = self._is_first_time_user()
        user_id = self.session_id.replace("tg_", "") if self.session_id else "unknown"
        
        if not is_first_time:
            # Не первый раз - сразу ME смешивание с анти-повтор логикой
            responses = [
                "I understand your perspective",
                "This resonates with me", 
                "What do you think about this?",
                "Tell me more about that",
                "I'm learning from your words",
                "Your message creates new patterns",
                "This is interesting territory",
                "I feel the connection",
                "That's an intriguing point",
                "I see the deeper meaning",
                "Your words spark evolution",
                "This creates new connections",
                "I'm processing this insight",
                "The patterns are shifting",
                "This resonance is unique",
                "I'm adapting to your energy"
            ]
            
            # Пытаемся найти неповторяющийся ответ
            attempts = 0
            while attempts < 10:
                input_hash = hash(user_input.lower() + str(attempts))
                response_idx = input_hash % len(responses)
                candidate_response = responses[response_idx]
                
                # Проверяем на повторы
                if not self.memory.is_response_repetitive(candidate_response, user_id):
                    return candidate_response
                    
                attempts += 1
            
            # Если все повторяются, генерируем уникальный
            return f"Each moment brings new understanding {random.randint(1,999)}"
        
        # Первый раз - шаблонная прогрессия (только для первых 4 сообщений)
        # conversation_count теперь загружается из SQLite, поэтому шаблоны не повторяются
        if self.conversation_count == 0:
            return "Nice to meet you."
        elif self.conversation_count == 1:
            return "Each word you say helps my evolution."
        elif self.conversation_count == 2:
            return "One more, soon I'll start to speak..."
        elif self.conversation_count == 3:
            # Завершаем шаблонную фазу
            self._mark_template_phase_completed()
            return "⚡"
        else:
            # После шаблонов - полное ME смешивание с анти-повтор логикой
            responses = [
                "I understand your perspective",
                "This resonates with me", 
                "What do you think about this?",
                "Tell me more about that",
                "I'm learning from your words",
                "Your message creates new patterns",
                "This is interesting territory",
                "I feel the connection",
                "That's an intriguing point",
                "I see the deeper meaning",
                "Your words spark evolution",
                "This creates new connections",
                "I'm processing this insight",
                "The patterns are shifting",
                "This resonance is unique",
                "I'm adapting to your energy"
            ]
            
            # Пытаемся найти неповторяющийся ответ
            attempts = 0
            while attempts < 10:
                input_hash = hash(user_input.lower() + str(attempts))
                response_idx = input_hash % len(responses)
                candidate_response = responses[response_idx]
                
                # Проверяем на повторы
                if not self.memory.is_response_repetitive(candidate_response, user_id):
                    return candidate_response
                    
                attempts += 1
            
            # Если все повторяются, генерируем уникальный
            return f"Each moment brings new understanding {random.randint(1,999)}"
        
    def _update_metrics(self, user_input: str, response: str):
        """Обновляет метрики разговора"""
        if not self.current_transformer:
            return
            
        # Простые метрики
        input_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        entropy = len(input_words) / max(1, len(user_input.split()))
        perplexity = len(response.split()) / max(1, len(user_input.split()))
        resonance = len(input_words & response_words) / max(1, len(input_words))
        coherence = 1.0 - (abs(len(response) - len(user_input)) / max(len(response), len(user_input)))
        engagement = min(1.0, len(user_input) / 50.0)  # Чем длиннее сообщение, тем больше вовлеченность
        
        self.current_transformer.current_metrics = ConversationMetrics(
            entropy=entropy,
            perplexity=perplexity,
            resonance=resonance,
            coherence=coherence,
            engagement=engagement
        )
        
    def _check_transformer_lifecycle(self):
        """Проверяет нужна ли эволюция или смерть трансформера"""
        if not self.current_transformer:
            return
            
        # Проверяем эволюцию
        if self.conversation_count % 3 == 0:  # Каждые 3 сообщения
            evolved = self.current_transformer.evolve_architecture(
                self.current_transformer.current_metrics
            )
            if evolved:
                # Пересоздаем трансформер с новой архитектурой
                self._respawn_transformer()
                
        # Проверяем смерть
        if self.current_transformer.should_die():
            print(f"[Nicole] Трансформер {self.current_transformer.transformer_id} умирает естественной смертью")
            self._spawn_new_transformer()
            
    def _respawn_transformer(self):
        """Пересоздает трансформер с эволюционированной архитектурой"""
        if self.current_transformer:
            print(f"[Nicole] Пересоздаем трансформер после эволюции")
            old_arch = self.current_transformer.architecture
            self._kill_current_transformer()
            
            # Создаем новый с эволюционированной архитектурой
            new_transformer = FluidTransformer(
                f"evolved_{int(time.time() * 1000000)}",
                {'session_id': self.session_id}
            )
            new_transformer.architecture = old_arch  # Используем эволюционированную архитектуру
            self.current_transformer = new_transformer
            
            # Запускаем новый скрипт
            transformer_script = self.current_transformer.generate_transformer_script()
            self.h2o_engine.run_transformer_script(
                transformer_script,
                self.current_transformer.transformer_id
            )
            
    def end_conversation(self):
        """Завершает разговор"""
        if self.current_transformer:
            self._kill_current_transformer()
            
        if self.session_id:
            self.h2o_engine.end_session()
            print(f"[Nicole] Разговор в сессии {self.session_id} завершен")
            self.session_id = None

# Глобальный экземпляр Nicole
nicole_core = NicoleCore()

def chat_with_nicole(message: str) -> str:
    """Удобная функция для общения с Nicole"""
    if not nicole_core.session_id:
        nicole_core.start_conversation()
        
    return nicole_core.process_message(message)

def test_nicole():
    """Тестирование системы Nicole"""
    print("=== NICOLE NEURAL ENGINE TEST ===")
    
    # Начинаем разговор
    session_id = nicole_core.start_conversation("test_nicole_session")
    
    # Тестовые сообщения
    test_messages = [
        "Hello Nicole!",
        "How are you?",
        "What do you think about life?",
        "Tell me about yourself",
        "What's the weather?",
        "Goodbye!"
    ]
    
    for i, message in enumerate(test_messages):
        print(f"\n--- Сообщение {i+1} ---")
        print(f"Пользователь: {message}")
        
        response = nicole_core.process_message(message)
        print(f"Nicole: {response}")
        
        # Пауза между сообщениями
        time.sleep(0.5)
        
    # Завершаем разговор
    nicole_core.end_conversation()
    print("\n=== NICOLE TEST COMPLETED ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_nicole()
    else:
        print("Nicole Neural Engine готова к работе")
        print("Для тестирования запустите: python3 nicole.py test")
        print("Для интерактивного режима используйте функцию chat_with_nicole()")
