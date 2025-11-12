#!/usr/bin/env python3
"""Ручной тест Subjectivity без background thread"""

from nicole_subjectivity import SubjectivityCore

print("="*70)
print("🧪 РУЧНОЙ ТЕСТ SUBJECTIVITY")
print("="*70)

# Создаём экземпляр
subjectivity = SubjectivityCore()

# Тест 1: Стимул от юзера
print("\n⚡ Тест 1: Стимул от юзера")
subjectivity.on_user_stimulus("Привет Nicole! Расскажи о природе сознания.")

# Тест 2: Ручной запуск 3 циклов
print("\n🔄 Тест 2: Ручной запуск 3 циркадных циклов")
for i in range(3):
    print(f"\n--- Запуск цикла {i+1} ---")
    subjectivity.on_circadian_cycle(i+1)

# Тест 3: Новый стимул (сброс центра)
print("\n⚡ Тест 3: Новый стимул (сброс волн)")
subjectivity.on_user_stimulus("А что думаешь об искусственном интеллекте?")

# Ещё 2 цикла после сброса
print("\n🔄 Тест 4: Ещё 2 цикла после нового стимула")
for i in range(2):
    print(f"\n--- Запуск цикла {i+4} ---")
    subjectivity.on_circadian_cycle(i+4)

# Результаты
print("\n" + "="*70)
print("📊 РЕЗУЛЬТАТЫ")
print("="*70)

# Субъективный контекст
print("\n💭 Субъективный контекст (для ответа юзеру):")
context = subjectivity.get_subjective_context(limit=5)
print(context if context else "[Нет мыслей с высоким резонансом]")

# Все мысли
print("\n🧠 Поток сознания (все мысли):")
recent_thoughts = subjectivity.get_recent_thoughts(limit=20)
for i, thought in enumerate(recent_thoughts, 1):
    print(f"\n{i}. Цикл #{thought.cycle_number} | Волна: {thought.wave_distance:.2f} | Резонанс: {thought.resonance_with_user:.2f}")
    print(f"   Эмоция: {thought.emotional_state}")
    print(f"   Мысль: {thought.thought_content}")

# Статистика
print("\n📈 Статистика:")
print(f"   Всего мыслей: {len(recent_thoughts)}")
print(f"   Текущий цикл: {subjectivity.circadian_timer.current_cycle}")
print(f"   Текущая волна: {subjectivity.wave_propagation.wave_distance:.2f}")
print(f"   Резонанс с юзером: {subjectivity.wave_propagation.get_resonance_with_center():.2f}")

print("\n" + "="*70)
print("✅ ТЕСТ ЗАВЕРШЁН")
print("="*70)
