#!/usr/bin/env python3
"""
Script for migrating existing Nicole databases to WAL mode
Run once to optimize all databases
"""

import os
import sys
from db_utils import optimize_database, get_db_stats

def main():
    print("=" * 60)
    print("Nicole Database Migration - WAL Mode + Indexes")
    print("=" * 60)

    # List of Nicole databases
    databases = [
        "nicole_memory.db",
        # Add other databases if used
    ]

    # Find existing databases
    existing_dbs = [db for db in databases if os.path.exists(db)]

    if not existing_dbs:
        print("\nNo databases found. They will be created with WAL mode on first use.")
        return

    print(f"\nFound {len(existing_dbs)} database(s) to migrate:\n")

    # Show current state
    for db in existing_dbs:
        stats = get_db_stats(db)
        print(f"📊 {db}:")
        print(f"   Size: {stats['size_mb']:.2f} MB")
        print(f"   Journal Mode: {stats['journal_mode']}")
        print(f"   Indexes: {stats['index_count']}")
        print(f"   Tables: {len(stats['tables'])}")
        print()

    # Ask for confirmation
    if len(sys.argv) < 2 or sys.argv[1] != "--yes":
        response = input("Proceed with migration? (yes/no): ").strip().lower()
        if response != "yes":
            print("Migration cancelled.")
            return

    print("\n" + "=" * 60)
    print("Starting migration...")
    print("=" * 60 + "\n")

    # Migrate each database
    for db in existing_dbs:
        print(f"\n🔄 Migrating {db}...")
        try:
            optimize_database(db)
            print(f"✅ {db} migration complete")
        except Exception as e:
            print(f"❌ Error migrating {db}: {e}")

    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60 + "\n")

    # Show new state
    print("After migration:\n")
    for db in existing_dbs:
        stats = get_db_stats(db)
        print(f"📊 {db}:")
        print(f"   Size: {stats['size_mb']:.2f} MB")
        print(f"   Journal Mode: {stats['journal_mode']}")
        print(f"   Indexes: {stats['index_count']}")
        print()

    print("🚀 Nicole Memory is now optimized!")
    print("\nExpected improvements:")
    print("  - 2-3x faster write operations")
    print("  - 2-5x faster queries (with indexes)")
    print("  - Better concurrency (readers don't block)")

if __name__ == "__main__":
    main()
