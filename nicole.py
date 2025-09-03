#!/usr/bin/env python3
"""
Nicole - Neural Organism Intelligence Conversational Language Engine
Флюидная нейронка без весов, создающая уникальные трансформеры для каждого диалога.
Посвящается Лео.
"""

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

@dataclass
class ConversationMetrics:
    """Метрики текущего разговора"""
    entropy: float = 0.0
    perplexity: float = 0.0
    resonance: float = 0.0
    coherence: float = 0.0
    engagement: float = 0.0
    
class NicoleMemory:
    """Система памяти Nicole без весов"""
    
    def __init__(self, db_path: str = "nicole_memory.db"):
        self.db_path = db_path
        self.init_database()
        
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
        """Обрабатывает сообщение пользователя"""
        with self.lock:
            if not self.current_transformer:
                self._spawn_new_transformer()
                
            # Простой ответ (пока что)
            response = self._generate_simple_response(user_input)
            
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
            
            self.conversation_count += 1
            return response
                
    def _generate_simple_response(self, user_input: str) -> str:
        """Генерирует простой ответ (временная заглушка)"""
        responses = [
            "Интересно, расскажи больше",
            "Понимаю тебя",
            "Да, это важная тема", 
            "Хм, нужно подумать",
            "Согласна с тобой",
            "А что ты об этом думаешь?",
            "Это сложный вопрос",
            "Мне нравится твой подход"
        ]
        
        # Простая эвристика выбора ответа
        input_hash = hash(user_input.lower())
        response_idx = input_hash % len(responses)
        
        return responses[response_idx]
        
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
        "Привет Nicole!",
        "Как дела?",
        "Что ты думаешь о смысле жизни?",
        "Расскажи мне о себе",
        "Какая погода?",
        "Пока!"
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