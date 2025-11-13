#!/usr/bin/env python3
"""
Database Utilities - Optimized SQLite connections
Includes WAL mode, indexes, and other optimizations for Nicole
"""

import sqlite3
from typing import Optional
import os


def get_optimized_connection(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Creates optimized SQLite connection with WAL mode

    Optimizations:
    - WAL mode (Write-Ahead Logging) - 2-3x faster writes
    - synchronous=NORMAL - safe + fast
    - cache_size=10000 - more cache = fewer disk operations
    - temp_store=MEMORY - temporary data in RAM
    - mmap_size - memory-mapped I/O for large DBs

    Args:
        db_path: Path to database file
        timeout: Lock wait timeout (seconds)

    Returns:
        Optimized SQLite connection
    """
    conn = sqlite3.connect(db_path, timeout=timeout)
    cursor = conn.cursor()

    # WAL mode - CRITICAL for performance
    # Write-Ahead Logging allows reading during writes
    cursor.execute("PRAGMA journal_mode=WAL")

    # synchronous=NORMAL - balance speed/safety
    # In WAL mode this is safe even during power failure
    cursor.execute("PRAGMA synchronous=NORMAL")

    # Increase cache to 10MB (10000 pages of 1KB)
    cursor.execute("PRAGMA cache_size=10000")

    # Temporary tables in memory
    cursor.execute("PRAGMA temp_store=MEMORY")

    # Memory-mapped I/O for large databases (30MB)
    cursor.execute("PRAGMA mmap_size=30000000")

    # Enable foreign keys (if used)
    cursor.execute("PRAGMA foreign_keys=ON")

    cursor.close()

    return conn


def create_memory_indexes(conn: sqlite3.Connection):
    """
    Creates indexes for nicole_memory.db if they don't exist yet

    Indexes are critical for search performance:
    - conversations: timestamp, session_id
    - associations: source_concept, strength
    - concepts: name (for fast lookup)
    """
    cursor = conn.cursor()

    # Check existing indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row[0] for row in cursor.fetchall()}

    # Indexes for conversations table
    indexes = [
        ("idx_conversations_timestamp", "conversations", "timestamp DESC"),
        ("idx_conversations_session", "conversations", "session_id"),
        ("idx_conversations_user_input", "conversations", "user_input"),
    ]

    # Indexes for associations (if table exists)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='associations'")
    if cursor.fetchone():
        indexes.extend([
            ("idx_associations_source", "associations", "source_concept"),
            ("idx_associations_target", "associations", "target_concept"),
            ("idx_associations_strength", "associations", "strength DESC"),
        ])

    # Indexes for concepts (if table exists)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
    if cursor.fetchone():
        indexes.append(("idx_concepts_name", "concepts", "name"))

    # Create missing indexes
    created = 0
    for index_name, table_name, columns in indexes:
        if index_name not in existing_indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({columns})")
                created += 1
                print(f"[db_utils] Created index: {index_name}")
            except sqlite3.OperationalError as e:
                # Table may not exist - that's normal
                pass

    if created > 0:
        conn.commit()
        print(f"[db_utils] Created {created} indexes for performance")

    cursor.close()


def optimize_database(db_path: str):
    """
    Optimizes existing database

    - Converts to WAL mode
    - Creates indexes
    - Runs VACUUM for defragmentation
    - Runs ANALYZE to update statistics
    """
    print(f"[db_utils] Optimizing database: {db_path}")

    if not os.path.exists(db_path):
        print(f"[db_utils] Database doesn't exist yet: {db_path}")
        return

    conn = get_optimized_connection(db_path)

    # Create indexes
    create_memory_indexes(conn)

    # ANALYZE - update statistics for query planner
    print("[db_utils] Running ANALYZE...")
    conn.execute("ANALYZE")

    # VACUUM - defragmentation (CAUTION: can be slow for large DBs)
    # Uncomment if database is heavily fragmented:
    # print("[db_utils] Running VACUUM...")
    # conn.execute("VACUUM")

    conn.commit()
    conn.close()

    print(f"[db_utils] Optimization complete: {db_path}")


def get_db_stats(db_path: str) -> dict:
    """Returns database statistics"""
    if not os.path.exists(db_path):
        return {"exists": False}

    conn = get_optimized_connection(db_path)
    cursor = conn.cursor()

    stats = {"exists": True}

    # File size
    stats["size_mb"] = os.path.getsize(db_path) / (1024 * 1024)

    # Journal mode
    cursor.execute("PRAGMA journal_mode")
    stats["journal_mode"] = cursor.fetchone()[0]

    # Page count
    cursor.execute("PRAGMA page_count")
    stats["page_count"] = cursor.fetchone()[0]

    # Page size
    cursor.execute("PRAGMA page_size")
    stats["page_size"] = cursor.fetchone()[0]

    # Table list
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    stats["tables"] = [row[0] for row in cursor.fetchall()]

    # Index count
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
    stats["index_count"] = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return stats


# Auto-optimization on module import
def auto_optimize_if_needed():
    """
    Automatically optimizes Nicole databases on first run
    """
    databases = [
        "nicole_memory.db",
        # Add other databases if needed
    ]

    for db in databases:
        if os.path.exists(db):
            stats = get_db_stats(db)
            if stats.get("journal_mode") != "wal":
                print(f"[db_utils] Auto-optimizing {db}...")
                optimize_database(db)


# Run auto-optimization on import
# Uncomment for automatic optimization:
# auto_optimize_if_needed()


if __name__ == "__main__":
    """Command-line utility for database optimization"""
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
