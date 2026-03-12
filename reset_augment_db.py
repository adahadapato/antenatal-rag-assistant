# reset_augment_db.py - Safe database reset script (STRICT PDF-FAITHFUL)
import sqlite3
import Initialize
from pathlib import Path

db_path = Initialize.augmented_db_path

# Verify database file exists
if not Path(db_path).exists():
    print(f"⚠️  Database not found at {db_path}")
    print("💡 Run setup_db_schema.py first to create the database.")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys (ensure constraints are respected)
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Get list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"📊 Found {len(tables)} tables: {tables}")
    
    # Define delete order (child tables first, then parent tables)
    # Matches our strict PDF-faithful schema
    delete_order = [
        "recommendations",    # Child: references conditions
        "tests",              # Child: references conditions
        "ultrasound",         # Child: references conditions
        "arguments",          # Child: references conditions
        "conditions",         # Parent: no foreign keys
    ]
    
    # Delete from each table (skip if table doesn't exist)
    for table in delete_order:
        if table in tables:
            cursor.execute(f"DELETE FROM {table}")
            print(f"✓ Cleared table: {table}")
        else:
            print(f"⚠️  Table not found: {table}")
    
    # Reset auto-increment counters (if using INTEGER PRIMARY KEY)
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('conditions', 'arguments', 'recommendations', 'tests', 'ultrasound');")
    print("✓ Reset auto-increment counters")
    
    conn.commit()
    print("\n✅ All argumentation tables cleared successfully.")
    
except sqlite3.Error as e:
    print(f"\n❌ Database error: {e}")
    conn.rollback()
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    conn.rollback()
finally:
    if conn:
        conn.close()
    print("🔒 Database connection closed.")