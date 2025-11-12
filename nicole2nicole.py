#!/usr/bin/env python3
"""
Nicole2Nicole - Модуль дообучения без весов
Асинхронное дообучение при смерти/перерождении трансформеров.
Учится на логах разговоров и эволюции архитектур.
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
# import nicole  # Убираем циклический импорт

@dataclass 
class LearningPattern:
    """Паттерн обучения из истории"""
    input_pattern: str
    output_pattern: str
    metrics_context: Dict
    architecture_context: Dict
    success_score: float
    frequency: int = 1
    
class Nicole2NicoleCore:
    """Ядро системы дообучения"""
    
    def __init__(self, memory_db: str = "nicole_memory.db", knowledge_file: str = "nicole_learned_knowledge.json"):
        self.memory_db = memory_db
        self.knowledge_file = knowledge_file  # ОПТИМИЗАЦИЯ: Auto-save файл
        self.learning_patterns = {}
        self.evolution_patterns = {}
        self.architecture_preferences = {}
        self.learning_lock = threading.Lock()
        self.is_learning = False
        self.learning_cycles = 0  # ОПТИМИЗАЦИЯ: Счетчик циклов для auto-save

        # ОПТИМИЗАЦИЯ: Auto-load знаний при старте
        if os.path.exists(self.knowledge_file):
            print(f"[Nicole2Nicole] Loading previous knowledge from {self.knowledge_file}...")
            self.import_learned_knowledge(self.knowledge_file)
        
    def analyze_conversation_logs(self, limit: int = 1000) -> List[LearningPattern]:
        """Анализирует логи разговоров для извлечения паттернов"""
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
                
                # Создаем паттерн обучения
                pattern = LearningPattern(
                    input_pattern=self._extract_input_pattern(user_input),
                    output_pattern=self._extract_output_pattern(nicole_output),
                    metrics_context=metrics,
                    architecture_context=config,
                    success_score=self._calculate_success_score(metrics)
                )
                
                patterns.append(pattern)
                
            except Exception as e:
                print(f"[Nicole2Nicole] Ошибка анализа лога: {e}")
                
        return patterns
        
    def _extract_input_pattern(self, user_input: str) -> str:
        """Извлекает паттерн из пользовательского ввода"""
        words = user_input.lower().split()
        
        # Категоризация типов сообщений
        if any(word in words for word in ['привет', 'hello', 'hi', 'здравствуй']):
            return "greeting"
        elif any(word in words for word in ['пока', 'bye', 'goodbye', 'увидимся']):
            return "farewell"  
        elif any(word in words for word in ['как', 'дела', 'жизнь', 'настроение']):
            return "status_inquiry"
        elif any(word in words for word in ['что', 'расскажи', 'объясни', 'опиши']):
            return "information_request"
        elif any(word in words for word in ['почему', 'зачем', 'отчего']):
            return "reasoning_request"
        elif len(words) > 10:
            return "long_message"
        elif len(words) < 3:
            return "short_message"
        else:
            return "general_conversation"
            
    def _extract_output_pattern(self, nicole_output: str) -> str:
        """Извлекает паттерн из ответа Nicole"""
        words = nicole_output.lower().split()
        
        if any(word in words for word in ['интересно', 'расскажи', 'больше']):
            return "curiosity_response"
        elif any(word in words for word in ['понимаю', 'согласна', 'да']):
            return "agreement_response"
        elif any(word in words for word in ['думать', 'сложный', 'вопрос']):
            return "contemplative_response"
        elif any(word in words for word in ['что', 'думаешь', 'мнение']):
            return "question_back"
        else:
            return "general_response"
            
    def _calculate_success_score(self, metrics: Dict) -> float:
        """Рассчитывает оценку успешности взаимодействия"""
        try:
            entropy = metrics.get('entropy', 0.5)
            perplexity = metrics.get('perplexity', 1.0)
            resonance = metrics.get('resonance', 0.3)
            coherence = metrics.get('coherence', 0.5)
            engagement = metrics.get('engagement', 0.5)
            
            # Взвешенная оценка успешности
            score = (
                entropy * 0.2 +
                (1.0 / max(0.1, perplexity)) * 0.3 +
                resonance * 0.3 +
                coherence * 0.1 +
                engagement * 0.1
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return 0.5  # Нейтральная оценка при ошибке
            
    def learn_from_patterns(self, patterns: List[LearningPattern]):
        """Обучается на паттернах взаимодействий"""
        with self.learning_lock:
            self.is_learning = True
            
            try:
                # Группируем паттерны по типам
                pattern_groups = {}
                for pattern in patterns:
                    key = (pattern.input_pattern, pattern.output_pattern)
                    if key not in pattern_groups:
                        pattern_groups[key] = []
                    pattern_groups[key].append(pattern)
                    
                # Анализируем каждую группу
                for (input_type, output_type), group in pattern_groups.items():
                    self._analyze_pattern_group(input_type, output_type, group)
                    
                # Анализируем эволюции архитектур
                self._analyze_architecture_evolution(patterns)
                
                print(f"[Nicole2Nicole] Обучение завершено на {len(patterns)} паттернах")
                
            finally:
                self.is_learning = False
                
    def _analyze_pattern_group(self, input_type: str, output_type: str, patterns: List[LearningPattern]):
        """Анализирует группу похожих паттернов"""
        if len(patterns) < 2:
            return
            
        # Вычисляем средние метрики для этого типа взаимодействия
        avg_metrics = {}
        metric_keys = ['entropy', 'perplexity', 'resonance', 'coherence', 'engagement']
        
        for key in metric_keys:
            values = []
            for pattern in patterns:
                if key in pattern.metrics_context:
                    values.append(pattern.metrics_context[key])
            if values:
                avg_metrics[key] = sum(values) / len(values)
                
        # Находим лучшие архитектуры для этого типа
        best_patterns = sorted(patterns, key=lambda p: p.success_score, reverse=True)[:3]
        best_architectures = [p.architecture_context for p in best_patterns]
        
        # Сохраняем паттерн
        pattern_key = f"{input_type}::{output_type}"
        self.learning_patterns[pattern_key] = {
            'avg_metrics': avg_metrics,
            'best_architectures': best_architectures,
            'frequency': len(patterns),
            'avg_success': sum(p.success_score for p in patterns) / len(patterns)
        }
        
        print(f"[Nicole2Nicole] Паттерн {pattern_key}: {len(patterns)} примеров, успешность {self.learning_patterns[pattern_key]['avg_success']:.3f}")
        
    def _analyze_architecture_evolution(self, patterns: List[LearningPattern]):
        """Анализирует эволюцию архитектур"""
        # Группируем по параметрам архитектуры
        arch_performance = {}
        
        for pattern in patterns:
            arch = pattern.architecture_context
            for param, value in arch.items():
                if param not in arch_performance:
                    arch_performance[param] = []
                arch_performance[param].append((value, pattern.success_score))
                
        # Находим оптимальные диапазоны для каждого параметра
        for param, value_scores in arch_performance.items():
            if len(value_scores) < 5:
                continue
                
            # Сортируем по успешности
            sorted_values = sorted(value_scores, key=lambda x: x[1], reverse=True)
            top_20_percent = sorted_values[:max(1, len(sorted_values) // 5)]
            
            # Находим диапазон лучших значений
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
                
        print(f"[Nicole2Nicole] Проанализированы предпочтения архитектуры для {len(self.architecture_preferences)} параметров")
        
    def suggest_architecture_improvements(self, current_arch: Dict, conversation_context: str) -> Dict:
        """
        Предлагает улучшения архитектуры на основе обучения + exploration noise

        ANTI-OVERFITTING: добавляет 10% шанс случайного исследования
        """
        if not self.learning_patterns or not self.architecture_preferences:
            return current_arch

        improved_arch = current_arch.copy()

        # EXPLORATION NOISE: 10% шанс случайного исследования
        # Предотвращает застревание в локальном оптимуме
        exploration_probability = 0.1

        if random.random() < exploration_probability:
            # Случайно выбираем параметр для исследования
            explorable_params = [p for p in improved_arch.keys()
                               if isinstance(improved_arch[p], (int, float))]

            if explorable_params:
                param_to_explore = random.choice(explorable_params)
                current_value = improved_arch[param_to_explore]

                # Случайное возмущение ±20%
                noise_factor = random.uniform(0.8, 1.2)
                improved_arch[param_to_explore] = current_value * noise_factor

                print(f"[Nicole2Nicole:Exploration] 🎲 Исследование параметра '{param_to_explore}': "
                      f"{current_value:.3f} → {improved_arch[param_to_explore]:.3f}")

        # Применяем изученные предпочтения
        for param, preferences in self.architecture_preferences.items():
            if param in improved_arch:
                current_val = improved_arch[param]
                
                # Двигаемся к оптимальному диапазону
                if isinstance(current_val, (int, float)):
                    target_val = preferences['avg']
                    
                    # Плавное движение к цели
                    if current_val < preferences['min']:
                        improved_arch[param] = min(preferences['max'], current_val * 1.1)
                    elif current_val > preferences['max']:
                        improved_arch[param] = max(preferences['min'], current_val * 0.9)
                    else:
                        # Мелкие случайные изменения в оптимальном диапазоне
                        noise = random.uniform(-0.05, 0.05)
                        improved_arch[param] = current_val * (1 + noise)
                        
        return improved_arch
        
    def suggest_response_strategy(self, user_input: str, current_metrics: Dict) -> str:
        """Предлагает стратегию ответа на основе изученных паттернов"""
        input_pattern = self._extract_input_pattern(user_input)
        
        # Ищем подходящие паттерны
        matching_patterns = []
        for pattern_key, pattern_data in self.learning_patterns.items():
            if pattern_key.startswith(input_pattern + "::"):
                matching_patterns.append((pattern_key, pattern_data))
                
        if not matching_patterns:
            return "general_response"
            
        # Выбираем лучший паттерн
        best_pattern = max(matching_patterns, key=lambda x: x[1]['avg_success'])
        output_pattern = best_pattern[0].split("::")[-1]
        
        return output_pattern
        
    def continuous_learning_loop(self):
        """
        Непрерывное обучение в фоновом режиме
        ОПТИМИЗАЦИЯ: Auto-save каждые 10 циклов (~5 минут)
        """
        while True:
            try:
                if not self.is_learning:
                    # Анализируем новые логи каждые 30 секунд
                    patterns = self.analyze_conversation_logs(100)
                    if patterns:
                        self.learn_from_patterns(patterns)
                        self.learning_cycles += 1

                        # ОПТИМИЗАЦИЯ: Auto-save каждые 10 циклов
                        if self.learning_cycles % 10 == 0:
                            print(f"[Nicole2Nicole] Auto-saving knowledge (cycle {self.learning_cycles})...")
                            self.export_learned_knowledge(self.knowledge_file)

                time.sleep(30)

            except Exception as e:
                print(f"[Nicole2Nicole:ERROR] Ошибка непрерывного обучения: {e}")
                time.sleep(60)  # Увеличиваем интервал при ошибке
                
    def start_continuous_learning(self):
        """Запускает непрерывное обучение в отдельном потоке"""
        learning_thread = threading.Thread(target=self.continuous_learning_loop, daemon=True)
        learning_thread.start()
        print("[Nicole2Nicole] Непрерывное обучение запущено")
        
    def get_learning_statistics(self) -> Dict:
        """Возвращает статистику обучения"""
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
        """Принудительно запускает сессию обучения"""
        print("[Nicole2Nicole] Принудительное обучение...")
        patterns = self.analyze_conversation_logs(500)
        if patterns:
            self.learn_from_patterns(patterns)
            return True
        return False
        
    def export_learned_knowledge(self, filepath: str):
        """Экспортирует изученные знания в файл"""
        knowledge = {
            'learning_patterns': self.learning_patterns,
            'evolution_patterns': self.evolution_patterns,
            'architecture_preferences': self.architecture_preferences,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
            
        print(f"[Nicole2Nicole] Знания экспортированы в {filepath}")
        
    def import_learned_knowledge(self, filepath: str):
        """Импортирует изученные знания из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
                
            self.learning_patterns.update(knowledge.get('learning_patterns', {}))
            self.evolution_patterns.update(knowledge.get('evolution_patterns', {}))
            self.architecture_preferences.update(knowledge.get('architecture_preferences', {}))
            
            print(f"[Nicole2Nicole] Знания импортированы из {filepath}")
            return True
            
        except Exception as e:
            print(f"[Nicole2Nicole:ERROR] Ошибка импорта: {e}")
            return False

# Интеграция с основной Nicole системой
class EnhancedFluidTransformer:  # Убираем наследование от nicole.FluidTransformer
    """Расширенный трансформер с дообучением"""
    
    def __init__(self, transformer_id: str, session_context: Dict = None, learning_core = None):
        # Убираем super() вызов
        self.transformer_id = transformer_id
        self.session_context = session_context or {}
        self.learning_core = learning_core
        
        # Применяем изученные улучшения архитектуры
        if self.learning_core:
            improved_arch = self.learning_core.suggest_architecture_improvements(
                self.architecture, 
                session_context.get('conversation_context', '')
            )
            self.architecture = improved_arch
            
    def evolve_architecture(self, metrics):
        """Эволюция с учетом изученных паттернов"""
        # Сначала стандартная эволюция
        evolved = super().evolve_architecture(metrics)
        
        # Затем применяем изученные улучшения
        if self.learning_core and evolved:
            learned_improvements = self.learning_core.suggest_architecture_improvements(
                self.architecture,
                ""  # Контекст разговора
            )
            
            # Мягко применяем изученные улучшения
            for param, target_val in learned_improvements.items():
                if param in self.architecture and isinstance(target_val, (int, float)):
                    current_val = self.architecture[param]
                    # Движемся на 10% к изученному оптимуму
                    self.architecture[param] = current_val * 0.9 + target_val * 0.1
                    
        return evolved

class EnhancedNicoleCore:  # Убираем наследование
    """Расширенное ядро Nicole с дообучением"""
    
    def __init__(self):
        # Убираем super() - это отдельный класс
        self.learning_core = Nicole2NicoleCore()
        self.learning_core.start_continuous_learning()
        
    def _spawn_new_transformer(self):
        """Создает новый трансформер с дообучением"""
        transformer_id = f"learned_{self.session_id}_{int(time.time() * 1000000)}"
        
        # Убиваем старый трансформер если есть
        if self.current_transformer:
            self._kill_current_transformer()
            
        # Создаем новый с дообучением
        self.current_transformer = EnhancedFluidTransformer(
            transformer_id, 
            {'session_id': self.session_id},
            self.learning_core
        )
        
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
            
            print(f"[Nicole] Новый обученный трансформер {transformer_id} создан")
            
        except Exception as e:
            print(f"[Nicole:ERROR] Ошибка создания обученного трансформера: {e}")
            
    def _generate_simple_response(self, user_input: str) -> str:
        """Генерирует ответ с учетом изученных паттернов"""
        # Получаем рекомендацию от системы обучения
        suggested_strategy = self.learning_core.suggest_response_strategy(
            user_input, 
            self.current_transformer.current_metrics.__dict__ if self.current_transformer else {}
        )
        
        # Выбираем ответы в зависимости от изученной стратегии
        response_strategies = {
            'curiosity_response': [
                "Интересно, расскажи больше",
                "Это увлекательно, продолжай",
                "Хочу узнать подробности"
            ],
            'agreement_response': [
                "Понимаю тебя",
                "Согласна с тобой", 
                "Да, ты прав"
            ],
            'contemplative_response': [
                "Хм, нужно подумать",
                "Это сложный вопрос",
                "Интересная мысль"
            ],
            'question_back': [
                "А что ты об этом думаешь?",
                "Как ты к этому относишься?",
                "Какое у тебя мнение?"
            ],
            'general_response': [
                "Понятно",
                "Хорошо",
                "Ясно"
            ]
        }
        
        responses = response_strategies.get(suggested_strategy, response_strategies['general_response'])
        
        # Выбираем ответ с элементом случайности
        input_hash = hash(user_input.lower())
        response_idx = input_hash % len(responses)
        
        return responses[response_idx]

# Глобальный экземпляр Enhanced Nicole
enhanced_nicole = EnhancedNicoleCore()

def chat_with_enhanced_nicole(message: str) -> str:
    """Общение с обученной Nicole"""
    if not enhanced_nicole.session_id:
        enhanced_nicole.start_conversation()
        
    return enhanced_nicole.process_message(message)

def test_nicole2nicole():
    """Тестирование системы дообучения"""
    print("=== NICOLE2NICOLE LEARNING TEST ===")
    
    # Создаем тестовые данные для обучения
    test_session = enhanced_nicole.start_conversation("learning_test")
    
    # Симулируем несколько разговоров
    test_conversations = [
        ("Привет!", "Привет! Как дела?"),
        ("Как дела?", "Хорошо, спасибо!"),
        ("Что нового?", "Много интересного происходит"),
        ("Расскажи о себе", "Я - Nicole, нейронная сеть"),
        ("Пока!", "До встречи!")
    ]
    
    for user_msg, expected_response in test_conversations:
        response = enhanced_nicole.process_message(user_msg)
        print(f"Пользователь: {user_msg}")
        print(f"Nicole: {response}")
        time.sleep(0.1)
        
    enhanced_nicole.end_conversation()
    
    # Принудительно запускаем обучение
    print("\\n--- Запуск обучения ---")
    enhanced_nicole.learning_core.force_learning_session()
    
    # Показываем статистику
    stats = enhanced_nicole.learning_core.get_learning_statistics()
    print(f"\\nСтатистика обучения:")
    print(f"- Изученных паттернов: {stats['learned_patterns']}")
    print(f"- Предпочтений архитектуры: {stats['architecture_preferences']}")
    print(f"- Топ паттерны:")
    for pattern, success, freq in stats['top_patterns']:
        print(f"  {pattern}: успешность={success:.3f}, частота={freq}")
        
    print("\\n=== NICOLE2NICOLE TEST COMPLETED ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_nicole2nicole()
    else:
        print("Nicole2Nicole Learning Engine готов к работе")
        print("Для тестирования запустите: python3 nicole2nicole.py test")
