#!/usr/bin/env python3
"""
Database Utilities - Оптимизированные подключения к SQLite
Включает WAL mode, индексы, и другие оптимизации для Nicole
"""

import sqlite3
from typing import Optional
import os


def get_optimized_connection(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Создает оптимизированное подключение к SQLite с WAL mode

    Оптимизации:
    - WAL mode (Write-Ahead Logging) - 2-3x быстрее записи
    - synchronous=NORMAL - безопасно + быстро
    - cache_size=10000 - больше кеша = меньше дисковых операций
    - temp_store=MEMORY - временные данные в RAM
    - mmap_size - memory-mapped I/O для больших БД

    Args:
        db_path: Путь к файлу базы данных
        timeout: Таймаут ожидания блокировки (секунды)

    Returns:
        Оптимизированное подключение к SQLite
    """
    conn = sqlite3.connect(db_path, timeout=timeout)
    cursor = conn.cursor()

    # WAL mode - КРИТИЧНО для производительности
    # Write-Ahead Logging позволяет читать во время записи
    cursor.execute("PRAGMA journal_mode=WAL")

    # synchronous=NORMAL - баланс скорость/безопасность
    # В WAL mode это безопасно даже при сбое питания
    cursor.execute("PRAGMA synchronous=NORMAL")

    # Увеличиваем кеш до 10MB (10000 страниц по 1KB)
    cursor.execute("PRAGMA cache_size=10000")

    # Временные таблицы в памяти
    cursor.execute("PRAGMA temp_store=MEMORY")

    # Memory-mapped I/O для больших баз (30MB)
    cursor.execute("PRAGMA mmap_size=30000000")

    # Включаем foreign keys (если используются)
    cursor.execute("PRAGMA foreign_keys=ON")

    cursor.close()

    return conn


def create_memory_indexes(conn: sqlite3.Connection):
    """
    Создает индексы для nicole_memory.db если их еще нет

    Индексы критичны для производительности поиска:
    - conversations: timestamp, session_id
    - associations: source_concept, strength
    - concepts: name (для быстрого lookup)
    """
    cursor = conn.cursor()

    # Проверяем существующие индексы
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row[0] for row in cursor.fetchall()}

    # Индексы для conversations таблицы
    indexes = [
        ("idx_conversations_timestamp", "conversations", "timestamp DESC"),
        ("idx_conversations_session", "conversations", "session_id"),
        ("idx_conversations_user_input", "conversations", "user_input"),
    ]

    # Индексы для associations (если таблица существует)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='associations'")
    if cursor.fetchone():
        indexes.extend([
            ("idx_associations_source", "associations", "source_concept"),
            ("idx_associations_target", "associations", "target_concept"),
            ("idx_associations_strength", "associations", "strength DESC"),
        ])

    # Индексы для concepts (если таблица существует)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
    if cursor.fetchone():
        indexes.append(("idx_concepts_name", "concepts", "name"))

    # Создаем индексы которых нет
    created = 0
    for index_name, table_name, columns in indexes:
        if index_name not in existing_indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({columns})")
                created += 1
                print(f"[db_utils] Created index: {index_name}")
            except sqlite3.OperationalError as e:
                # Таблица может не существовать - это нормально
                pass

    if created > 0:
        conn.commit()
        print(f"[db_utils] Created {created} indexes for performance")

    cursor.close()


def optimize_database(db_path: str):
    """
    Оптимизирует существующую базу данных

    - Конвертирует в WAL mode
    - Создает индексы
    - Запускает VACUUM для дефрагментации
    - Запускает ANALYZE для обновления статистики
    """
    print(f"[db_utils] Optimizing database: {db_path}")

    if not os.path.exists(db_path):
        print(f"[db_utils] Database doesn't exist yet: {db_path}")
        return

    conn = get_optimized_connection(db_path)

    # Создаем индексы
    create_memory_indexes(conn)

    # ANALYZE - обновляем статистику для query planner
    print("[db_utils] Running ANALYZE...")
    conn.execute("ANALYZE")

    # VACUUM - дефрагментация (ОСТОРОЖНО: может быть медленным для больших БД)
    # Раскомментируйте если база сильно фрагментирована:
    # print("[db_utils] Running VACUUM...")
    # conn.execute("VACUUM")

    conn.commit()
    conn.close()

    print(f"[db_utils] Optimization complete: {db_path}")


def get_db_stats(db_path: str) -> dict:
    """Возвращает статистику базы данных"""
    if not os.path.exists(db_path):
        return {"exists": False}

    conn = get_optimized_connection(db_path)
    cursor = conn.cursor()

    stats = {"exists": True}

    # Размер файла
    stats["size_mb"] = os.path.getsize(db_path) / (1024 * 1024)

    # Режим журнала
    cursor.execute("PRAGMA journal_mode")
    stats["journal_mode"] = cursor.fetchone()[0]

    # Количество страниц
    cursor.execute("PRAGMA page_count")
    stats["page_count"] = cursor.fetchone()[0]

    # Размер страницы
    cursor.execute("PRAGMA page_size")
    stats["page_size"] = cursor.fetchone()[0]

    # Список таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    stats["tables"] = [row[0] for row in cursor.fetchall()]

    # Количество индексов
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
    stats["index_count"] = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return stats


# Автоматическая оптимизация при импорте модуля
def auto_optimize_if_needed():
    """
    Автоматически оптимизирует базы данных Nicole при первом запуске
    """
    databases = [
        "nicole_memory.db",
        # Добавьте другие базы если нужно
    ]

    for db in databases:
        if os.path.exists(db):
            stats = get_db_stats(db)
            if stats.get("journal_mode") != "wal":
                print(f"[db_utils] Auto-optimizing {db}...")
                optimize_database(db)


# Запускаем автооптимизацию при импорте
# Раскомментируйте для автоматической оптимизации:
# auto_optimize_if_needed()


if __name__ == "__main__":
    """Утилита командной строки для оптимизации баз"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 db_utils.py optimize <db_path>  - Optimize database")
        print("  python3 db_utils.py stats <db_path>     - Show database stats")
        print("  python3 db_utils.py auto                - Auto-optimize all Nicole databases")
        sys.exit(1)

    command = sys.argv[1]

    if command == "optimize" and len(sys.argv) == 3:
        optimize_database(sys.argv[2])
    elif command == "stats" and len(sys.argv) == 3:
        stats = get_db_stats(sys.argv[2])
        print("\n=== Database Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    elif command == "auto":
        auto_optimize_if_needed()
    else:
        print("Invalid command")
        sys.exit(1)
