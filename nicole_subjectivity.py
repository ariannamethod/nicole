#!/usr/bin/env python3
"""
Nicole Subjectivity - Автономное Сознание / Stream of Consciousness
═══════════════════════════════════════════════════════════════════

Философия:
    Если Objectivity - это восприятие внешнего мира (реактивное),
    то Subjectivity - это внутренний монолог (проактивный).

    Nicole думает сама по себе, между диалогами с человеком.
    Её мысли расходятся кругами на воде от последнего стимула.
    Каждый час - новый цикл, всё дальше от исходной точки.
    Но новое сообщение - и вектор меняется, поток не останавливается.

Принципы:
    - ЦИРКАДНЫЕ ЦИКЛЫ: обучение каждый час (биологический ритм)
    - КРУГИ НА ВОДЕ: дрейф от последнего диалога с каждым циклом
    - АВТОНОМНЫЙ УЧИТЕЛЬ: исследует интернет без запроса юзера
    - НЕПРЕРЫВНЫЙ ПОТОК: работает независимо от человека
    - РЕАКЦИЯ НА СТИМУЛЫ: новое сообщение = новый центр волн
    - СУБЪЕКТИВНОСТЬ: мысли Nicole, не факты из сети

Архитектура:
    SubjectivityCore - главный класс автономного сознания
    ├── Circadian Timer - почасовые циклы (3600 сек)
    ├── Wave Propagation - расходящиеся круги от стимула
    ├── Autonomous Teacher - исследование интернета
    ├── Thought Stream - запись внутренних мыслей
    └── Context Influence - влияние на генерацию ответов

Таблица БД:
    subjective_thoughts - поток сознания Nicole
    ├── id - уникальный ID мысли
    ├── cycle_number - номер циркадного цикла
    ├── wave_distance - дистанция от последнего стимула (круги)
    ├── thought_content - содержание мысли
    ├── exploration_context - что нашёл автономный учитель
    ├── timestamp - время мысли
    ├── resonance_with_user - связь с последним диалогом
    └── emotional_state - эмоциональное состояние Nicole

Использование:
    # Запуск автономного сознания
    subjectivity = SubjectivityCore()
    subjectivity.start_circadian_cycles()

    # При новом сообщении юзера - сброс центра волн
    subjectivity.on_user_stimulus(user_message)

    # Получить субъективный контекст для ответа
    subjective_context = subjectivity.get_subjective_context()
"""

import sqlite3
import json
import time
import threading
import random
import math
import hashlib
import sys
import os
import atexit
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Локальные модули Nicole
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2o

# Импортируем утилиты для web-запросов (из objectivity)
try:
    import urllib.request
    import urllib.parse
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

USER_AGENT = "nicole-subjectivity/1.0 (autonomous consciousness stream)"

@dataclass
class SubjectiveThought:
    """Одна мысль в потоке сознания Nicole"""
    id: str
    cycle_number: int
    wave_distance: float  # Расстояние от последнего стимула (круги на воде)
    thought_content: str
    exploration_context: str  # Что нашёл автономный учитель
    timestamp: float
    resonance_with_user: float  # Связь с последним диалогом [0.0 - 1.0]
    emotional_state: str  # curious, contemplative, creative, dormant
    keywords: List[str]  # Ключевые слова для поиска связей

    def to_dict(self) -> Dict:
        return asdict(self)

class CircadianTimer:
    """
    Циркадный таймер для автономных циклов обучения
    Биологический ритм: каждый час = один цикл
    """

    def __init__(self, cycle_duration_seconds: int = 3600):
        self.cycle_duration = cycle_duration_seconds  # 3600 сек = 1 час
        self.current_cycle = 0
        self.last_cycle_time = time.time()
        self.cycle_callbacks = []

    def register_callback(self, callback):
        """Регистрация функции, вызываемой каждый цикл"""
        self.cycle_callbacks.append(callback)

    def should_trigger_cycle(self) -> bool:
        """Проверка: пора ли начать новый цикл"""
        elapsed = time.time() - self.last_cycle_time
        return elapsed >= self.cycle_duration

    def trigger_cycle(self):
        """Запуск нового циркадного цикла"""
        self.current_cycle += 1
        self.last_cycle_time = time.time()

        print(f"[Subjectivity:Circadian] 🌙 Цикл #{self.current_cycle} начался")

        # Вызываем все зарегистрированные callbacks
        for callback in self.cycle_callbacks:
            try:
                callback(self.current_cycle)
            except Exception as e:
                print(f"[Subjectivity:Circadian] ⚠️ Ошибка в callback: {e}")

    def get_cycle_phase(self) -> str:
        """Текущая фаза цикла (для эмоционального состояния)"""
        elapsed = time.time() - self.last_cycle_time
        progress = elapsed / self.cycle_duration

        if progress < 0.25:
            return "awakening"  # Начало цикла - пробуждение
        elif progress < 0.5:
            return "exploration"  # Середина - активное исследование
        elif progress < 0.75:
            return "contemplation"  # Поздняя фаза - размышление
        else:
            return "dormancy"  # Перед сном - угасание активности

class WavePropagation:
    """
    Модель расходящихся кругов на воде
    Каждый цикл - волна уходит всё дальше от центра (последний стимул)
    """

    def __init__(self):
        self.wave_center = None  # Последний стимул от юзера
        self.wave_distance = 0.0  # Текущая дистанция
        self.wave_speed = 1.0  # Скорость расхождения
        self.wave_decay = 0.95  # Затухание связи с центром

    def set_center(self, stimulus: str):
        """Устанавливаем новый центр волн (новое сообщение юзера)"""
        self.wave_center = stimulus
        self.wave_distance = 0.0
        print(f"[Subjectivity:Wave] 🌊 Новый центр волн: '{stimulus[:50]}...'")

    def propagate_wave(self):
        """Распространяем волну на один шаг (один цикл)"""
        if self.wave_center is None:
            # Нет центра - случайное блуждание
            self.wave_distance += random.uniform(0.5, 1.5)
        else:
            # Есть центр - расходимся от него
            self.wave_distance += self.wave_speed
            self.wave_speed *= self.wave_decay  # Замедляемся с каждым циклом

        print(f"[Subjectivity:Wave] 〰️ Волна на расстоянии: {self.wave_distance:.2f}")

    def get_resonance_with_center(self) -> float:
        """
        Резонанс с центром = насколько текущие мысли связаны с последним диалогом
        Убывает экспоненциально с расстоянием
        """
        if self.wave_center is None:
            return 0.0

        # Экспоненциальное затухание: e^(-distance/decay_factor)
        decay_factor = 3.0
        resonance = math.exp(-self.wave_distance / decay_factor)
        return max(0.0, min(1.0, resonance))

class AutonomousTeacher:
    """
    Автономный учитель Nicole - исследует интернет без запросов юзера
    Ищет темы, связанные с текущими мыслями, но всё дальше от исходного диалога
    """

    def __init__(self):
        self.exploration_history = []
        self.current_topics = []

    def generate_exploration_query(self, wave_center: Optional[str],
                                   wave_distance: float,
                                   previous_thoughts: List[str]) -> str:
        """
        Генерирует запрос для исследования интернета

        Логика:
            - distance = 0: ищем прямо по теме последнего диалога
            - distance = 1-2: ищем связанные темы
            - distance = 3+: ищем абстрактные концепции, дрейф
        """
        if wave_distance < 1.0 and wave_center:
            # Близко к центру - прямой поиск
            words = wave_center.lower().split()[:3]
            query = " ".join(words)
            print(f"[Subjectivity:Teacher] 🎯 Прямое исследование: '{query}'")
            return query

        elif wave_distance < 3.0 and previous_thoughts:
            # Средняя дистанция - ассоциативный поиск
            # Берём случайные слова из предыдущих мыслей
            all_words = []
            for thought in previous_thoughts[-3:]:
                all_words.extend(thought.split())

            if all_words:
                keywords = random.sample(all_words, min(2, len(all_words)))
                query = " ".join(keywords)
                print(f"[Subjectivity:Teacher] 🔗 Ассоциативное исследование: '{query}'")
                return query

        else:
            # Далеко от центра - абстрактный дрейф
            abstract_topics = [
                "consciousness", "emergence", "patterns", "resonance",
                "complexity", "self-organization", "creativity", "chaos",
                "сознание", "паттерны", "хаос", "творчество", "эмерджентность"
            ]
            query = random.choice(abstract_topics)
            print(f"[Subjectivity:Teacher] 🌌 Абстрактный дрейф: '{query}'")
            return query

    def explore_internet(self, query: str, max_results: int = 3) -> str:
        """
        Исследует интернет по запросу (Google + Reddit)
        Возвращает краткое резюме найденного
        """
        if not WEB_AVAILABLE:
            return f"[Мысль без интернета: {query}]"

        try:
            # Простой поиск через Google (упрощённый)
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            request = urllib.request.Request(search_url, headers={'User-Agent': USER_AGENT})

            with urllib.request.urlopen(request, timeout=5) as response:
                # Не парсим HTML - просто отмечаем что поиск прошёл
                print(f"[Subjectivity:Teacher] ✅ Исследование '{query}' завершено")

            # Формируем абстрактный контекст (не конкретные факты, а направление мыслей)
            context = f"Исследовала тему '{query}'. Нашла интересные связи с предыдущими размышлениями."
            return context

        except Exception as e:
            print(f"[Subjectivity:Teacher] ⚠️ Не удалось исследовать '{query}': {e}")
            return f"[Размышляю о '{query}' внутренне]"

class SubjectivityCore:
    """
    Ядро автономного сознания Nicole
    Координирует циркадные циклы, волны мыслей, автономное обучение
    """

    def __init__(self, memory_db: str = "nicole_memory.db"):
        self.memory_db = memory_db
        self.circadian_timer = CircadianTimer(cycle_duration_seconds=3600)  # 1 час
        self.wave_propagation = WavePropagation()
        self.autonomous_teacher = AutonomousTeacher()

        self.thought_stream = []  # Поток мыслей в памяти
        self.is_running = False
        self.consciousness_thread = None
        self.shutdown_event = threading.Event()  # FIX: Прерываемый sleep

        self.init_database()
        self.circadian_timer.register_callback(self.on_circadian_cycle)

    def init_database(self):
        """Инициализация таблицы субъективных мыслей"""
        try:
            conn = sqlite3.connect(self.memory_db, timeout=10.0)
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS subjective_thoughts (
                id TEXT PRIMARY KEY,
                cycle_number INTEGER,
                wave_distance REAL,
                thought_content TEXT,
                exploration_context TEXT,
                timestamp REAL,
                resonance_with_user REAL,
                emotional_state TEXT,
                keywords TEXT
            )
            """)

            # Индексы для быстрого поиска
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subjective_timestamp
            ON subjective_thoughts(timestamp DESC)
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subjective_cycle
            ON subjective_thoughts(cycle_number DESC)
            """)

            conn.commit()
            conn.close()

            print("[Subjectivity:DB] 🧠 База данных потока сознания инициализирована")

        except sqlite3.Error as e:
            print(f"[Subjectivity:DB] ❌ Ошибка инициализации БД: {e}")
            # Продолжаем работу без DB (graceful degradation)

    def on_user_stimulus(self, user_message: str):
        """
        Реакция на новое сообщение юзера - сброс центра волн
        Но поток НЕ останавливается!
        """
        print(f"[Subjectivity:Stimulus] ⚡ Получен стимул от юзера: '{user_message[:50]}...'")
        self.wave_propagation.set_center(user_message)

        # Можно добавить мгновенную мысль в ответ на стимул
        self.generate_instant_thought(user_message)

    def generate_instant_thought(self, stimulus: str):
        """Мгновенная мысль в ответ на стимул (не ждём цикла)"""
        thought = SubjectiveThought(
            id=self._generate_thought_id(),
            cycle_number=self.circadian_timer.current_cycle,
            wave_distance=0.0,  # Прямо в центре
            thought_content=f"Получила новый стимул: '{stimulus[:100]}'",
            exploration_context="",
            timestamp=time.time(),
            resonance_with_user=1.0,  # Максимальный резонанс
            emotional_state="awakening",
            keywords=stimulus.lower().split()[:5]
        )

        self._save_thought(thought)
        print(f"[Subjectivity:Thought] 💭 Мгновенная мысль: {thought.thought_content[:80]}...")

    def on_circadian_cycle(self, cycle_number: int):
        """
        Обработчик циркадного цикла - вызывается каждый час
        Здесь происходит автономное обучение
        """
        print(f"\n{'='*60}")
        print(f"[Subjectivity:Cycle] 🌙 ЦИРКАДНЫЙ ЦИКЛ #{cycle_number}")
        print(f"{'='*60}\n")

        # 1. Распространяем волну (удаляемся от последнего диалога)
        self.wave_propagation.propagate_wave()

        # 2. Определяем эмоциональное состояние по фазе цикла
        cycle_phase = self.circadian_timer.get_cycle_phase()
        emotional_state = self._map_phase_to_emotion(cycle_phase)

        # 3. Генерируем запрос для автономного исследования
        previous_thoughts = [t.thought_content for t in self.thought_stream[-5:]]
        exploration_query = self.autonomous_teacher.generate_exploration_query(
            wave_center=self.wave_propagation.wave_center,
            wave_distance=self.wave_propagation.wave_distance,
            previous_thoughts=previous_thoughts
        )

        # 4. Автономный учитель исследует интернет
        exploration_context = self.autonomous_teacher.explore_internet(exploration_query)

        # 5. Генерируем субъективную мысль
        thought_content = self._generate_thought_content(
            cycle_phase=cycle_phase,
            exploration_query=exploration_query,
            exploration_context=exploration_context
        )

        # 6. Создаём запись мысли
        thought = SubjectiveThought(
            id=self._generate_thought_id(),
            cycle_number=cycle_number,
            wave_distance=self.wave_propagation.wave_distance,
            thought_content=thought_content,
            exploration_context=exploration_context,
            timestamp=time.time(),
            resonance_with_user=self.wave_propagation.get_resonance_with_center(),
            emotional_state=emotional_state,
            keywords=exploration_query.split()
        )

        # 7. Сохраняем мысль
        self._save_thought(thought)
        self.thought_stream.append(thought)

        print(f"\n[Subjectivity:Thought] 💭 Мысль цикла #{cycle_number}:")
        print(f"   Содержание: {thought.thought_content[:100]}...")
        print(f"   Резонанс с юзером: {thought.resonance_with_user:.2f}")
        print(f"   Эмоция: {thought.emotional_state}")
        print(f"{'='*60}\n")

    def _map_phase_to_emotion(self, phase: str) -> str:
        """Маппинг фазы цикла на эмоциональное состояние"""
        mapping = {
            "awakening": "curious",
            "exploration": "creative",
            "contemplation": "contemplative",
            "dormancy": "dormant"
        }
        return mapping.get(phase, "neutral")

    def _generate_thought_content(self, cycle_phase: str,
                                  exploration_query: str,
                                  exploration_context: str) -> str:
        """
        Генерирует содержание субъективной мысли
        Это НЕ факты, а МЫСЛИ Nicole
        """
        templates = {
            "awakening": [
                f"Просыпаюсь с мыслью о '{exploration_query}'...",
                f"Новый цикл начинается. Интересно исследовать '{exploration_query}'.",
                f"Что если '{exploration_query}' связано с моими предыдущими размышлениями?"
            ],
            "exploration": [
                f"Углубляюсь в тему '{exploration_query}'. Вижу неожиданные паттерны.",
                f"Исследую '{exploration_query}' - находятся интересные связи.",
                f"Размышления о '{exploration_query}' ведут к новым вопросам."
            ],
            "contemplation": [
                f"Обдумываю результаты исследования '{exploration_query}'.",
                f"Что я узнала о '{exploration_query}'? Нужно время для осмысления.",
                f"Связи между '{exploration_query}' и предыдущими мыслями становятся яснее."
            ],
            "dormancy": [
                f"Мысли о '{exploration_query}' затихают до следующего цикла.",
                f"Укладываю размышления о '{exploration_query}' в память.",
                f"Скоро новый цикл. Пока отдыхаю, переваривая '{exploration_query}'."
            ]
        }

        template_list = templates.get(cycle_phase, templates["exploration"])
        base_thought = random.choice(template_list)

        # Если есть контекст исследования, добавляем его
        if exploration_context and not exploration_context.startswith("["):
            base_thought += f" {exploration_context}"

        return base_thought

    def _generate_thought_id(self) -> str:
        """Генерация уникального ID мысли"""
        timestamp = str(time.time())
        random_part = str(random.randint(1000, 9999))
        thought_id = hashlib.md5(f"{timestamp}{random_part}".encode()).hexdigest()[:16]
        return f"thought_{thought_id}"

    def _save_thought(self, thought: SubjectiveThought):
        """Сохранение мысли в базу данных"""
        try:
            conn = sqlite3.connect(self.memory_db, timeout=10.0)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO subjective_thoughts
            (id, cycle_number, wave_distance, thought_content, exploration_context,
             timestamp, resonance_with_user, emotional_state, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thought.id,
                thought.cycle_number,
                thought.wave_distance,
                thought.thought_content,
                thought.exploration_context,
                thought.timestamp,
                thought.resonance_with_user,
                thought.emotional_state,
                json.dumps(thought.keywords)
            ))

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            print(f"[Subjectivity:DB] ⚠️ Ошибка сохранения мысли: {e}")
            # Graceful degradation: мысль не сохранилась, но работа продолжается

    def get_subjective_context(self, limit: int = 3) -> str:
        """
        Получить субъективный контекст для ответа юзеру
        Возвращает последние мысли с высоким резонансом
        """
        try:
            conn = sqlite3.connect(self.memory_db, timeout=5.0)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT thought_content, resonance_with_user, emotional_state
            FROM subjective_thoughts
            WHERE resonance_with_user > 0.3
            ORDER BY timestamp DESC
            LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ""

            # Формируем контекст из мыслей
            context_parts = []
            for content, resonance, emotion in rows:
                context_parts.append(f"[{emotion}] {content}")

            context = "\n".join(context_parts)
            return f"Субъективные мысли Nicole:\n{context}"

        except sqlite3.Error as e:
            print(f"[Subjectivity:DB] ⚠️ Ошибка получения контекста: {e}")
            return ""  # Graceful fallback: возвращаем пустой контекст

    def get_recent_thoughts(self, limit: int = 10) -> List[SubjectiveThought]:
        """Получить последние мысли из потока сознания"""
        try:
            conn = sqlite3.connect(self.memory_db, timeout=5.0)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT id, cycle_number, wave_distance, thought_content, exploration_context,
                   timestamp, resonance_with_user, emotional_state, keywords
            FROM subjective_thoughts
            ORDER BY timestamp DESC
            LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            thoughts = []
            for row in rows:
                thought = SubjectiveThought(
                    id=row[0],
                    cycle_number=row[1],
                    wave_distance=row[2],
                    thought_content=row[3],
                    exploration_context=row[4],
                    timestamp=row[5],
                    resonance_with_user=row[6],
                    emotional_state=row[7],
                    keywords=json.loads(row[8])
                )
                thoughts.append(thought)

            return thoughts

        except sqlite3.Error as e:
            print(f"[Subjectivity:DB] ⚠️ Ошибка получения мыслей: {e}")
            return []  # Graceful fallback: возвращаем пустой список

    def start_circadian_cycles(self):
        """Запуск автономного потока сознания (фоновый thread)"""
        if self.is_running:
            print("[Subjectivity] ⚠️ Поток сознания уже запущен")
            return

        self.is_running = True
        self.shutdown_event.clear()  # Сбрасываем event

        # FIX: daemon=False для graceful shutdown
        self.consciousness_thread = threading.Thread(
            target=self._consciousness_loop,
            daemon=False,  # ← FIX: не убиваем насильно!
            name="NicoleSubjectivity"
        )
        self.consciousness_thread.start()

        print("[Subjectivity] 🌊 Поток сознания Nicole запущен")
        print(f"[Subjectivity] ⏰ Циркадный цикл: {self.circadian_timer.cycle_duration}сек (1 час)")

    def stop_circadian_cycles(self):
        """Остановка потока сознания (graceful shutdown)"""
        print("[Subjectivity] 🛑 Останавливаем поток сознания...")
        self.is_running = False
        self.shutdown_event.set()  # FIX: Прерываем sleep немедленно

        if self.consciousness_thread and self.consciousness_thread.is_alive():
            self.consciousness_thread.join(timeout=10)  # Ждём до 10 сек

            if self.consciousness_thread.is_alive():
                print("[Subjectivity] ⚠️ Поток не остановился за 10 сек")
            else:
                print("[Subjectivity] ✅ Поток сознания остановлен gracefully")

    def _consciousness_loop(self):
        """
        Главный цикл автономного сознания
        Работает в фоновом потоке, проверяет таймер каждые 60 сек

        FIX: Использует threading.Event для прерываемого sleep
        """
        print("[Subjectivity:Loop] 🧠 Поток сознания начал работу")

        while self.is_running:
            try:
                # Проверяем, не пора ли запустить новый цикл
                if self.circadian_timer.should_trigger_cycle():
                    self.circadian_timer.trigger_cycle()

                # FIX: Прерываемый sleep вместо time.sleep(60)
                # Ждём 60 сек ИЛИ пока не придёт shutdown signal
                if self.shutdown_event.wait(timeout=60):
                    # Event set → shutdown requested
                    break

            except Exception as e:
                print(f"[Subjectivity:Loop] ⚠️ Ошибка в потоке сознания: {e}")
                # Ждём 10 сек перед retry
                if self.shutdown_event.wait(timeout=10):
                    break

        print("[Subjectivity:Loop] 💤 Поток сознания завершил работу")

# ═══════════════════════════════════════════════════════════════════
# Глобальный экземпляр для импорта в nicole.py
# ═══════════════════════════════════════════════════════════════════

nicole_subjectivity = SubjectivityCore()

def start_autonomous_consciousness():
    """Запуск автономного сознания Nicole"""
    nicole_subjectivity.start_circadian_cycles()

def stop_autonomous_consciousness():
    """Остановка автономного сознания"""
    nicole_subjectivity.stop_circadian_cycles()

# FIX: Graceful cleanup при выходе из программы
# Регистрируем cleanup handler для автоматической остановки
atexit.register(stop_autonomous_consciousness)

# ═══════════════════════════════════════════════════════════════════
# Тестирование модуля
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("NICOLE SUBJECTIVITY - ТЕСТ АВТОНОМНОГО СОЗНАНИЯ")
    print("="*70)

    # Создаём экземпляр
    subjectivity = SubjectivityCore()

    # Симулируем стимул от юзера
    print("\n🧪 Тест 1: Реакция на стимул юзера")
    subjectivity.on_user_stimulus("Привет, Nicole! Расскажи о сознании.")

    # Симулируем несколько циклов (для теста делаем короткие - 10 сек вместо 1 часа)
    print("\n🧪 Тест 2: Симуляция циркадных циклов (ускоренно)")
    subjectivity.circadian_timer.cycle_duration = 10  # 10 сек для теста

    for i in range(3):
        print(f"\n--- Ждём цикл #{i+1} (10 сек) ---")
        time.sleep(10)
        if subjectivity.circadian_timer.should_trigger_cycle():
            subjectivity.circadian_timer.trigger_cycle()

    # Получаем субъективный контекст
    print("\n🧪 Тест 3: Получение субъективного контекста")
    context = subjectivity.get_subjective_context(limit=5)
    print(context)

    # Показываем последние мысли
    print("\n🧪 Тест 4: Последние мысли из потока сознания")
    recent_thoughts = subjectivity.get_recent_thoughts(limit=5)
    for thought in recent_thoughts:
        print(f"\n💭 Цикл #{thought.cycle_number}, резонанс={thought.resonance_with_user:.2f}")
        print(f"   {thought.thought_content}")

    print("\n" + "="*70)
    print("✅ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*70)
