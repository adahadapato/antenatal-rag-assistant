# setup_augment_db_schema.py - Create correct database schema (STRICT PDF-FAITHFUL)
import sqlite3
import Initialize
from pathlib import Path

db_path = Initialize.augmented_db_path

# Verify database file exists
if not Path(db_path).exists():
    print(f"⚠️  Database not found at {db_path}")
    print("💡 Creating new database...")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")

# Drop existing tables (in correct order to respect foreign keys)
print("🗑️  Dropping existing tables...")
cursor.execute("DROP TABLE IF EXISTS reasons")
cursor.execute("DROP TABLE IF EXISTS ultrasound")
cursor.execute("DROP TABLE IF EXISTS tests")
cursor.execute("DROP TABLE IF EXISTS recommendations")
cursor.execute("DROP TABLE IF EXISTS arguments")
cursor.execute("DROP TABLE IF EXISTS conditions")

# Create conditions table
print("📋 Creating conditions table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS conditions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT UNIQUE,
        condition_name TEXT NOT NULL,
        criteria TEXT,
        source_page INTEGER,
        source_section TEXT,
        source_reference_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Create arguments table (strict PDF-faithful - NO warrant/rebuttal)
# UPDATED: Added guideline_reference column
print("📋 Creating arguments table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS arguments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT,
        arg_id TEXT,
        claim TEXT NOT NULL,
        timing TEXT,
        category TEXT,
        source_column TEXT,
        guideline_reference TEXT,
        FOREIGN KEY (condition_id) REFERENCES conditions(condition_id)
    )
""")

# Create recommendations table (strict PDF-faithful)
# UPDATED: Added guideline_reference column
print("📋 Creating recommendations table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT,
        claim TEXT NOT NULL,
        timing TEXT,
        category TEXT,
        source_column TEXT,
        guideline_reference TEXT,
        FOREIGN KEY (condition_id) REFERENCES conditions(condition_id)
    )
""")

# Create tests table (strict PDF-faithful)
# UPDATED: Added guideline_reference column
print("📋 Creating tests table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT,
        claim TEXT NOT NULL,
        timing TEXT,
        category TEXT,
        source_column TEXT,
        guideline_reference TEXT,
        FOREIGN KEY (condition_id) REFERENCES conditions(condition_id)
    )
""")

# Create ultrasound table (strict PDF-faithful)
# UPDATED: Added guideline_reference column
print("📋 Creating ultrasound table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ultrasound (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT,
        claim TEXT NOT NULL,
        timing TEXT,
        category TEXT,
        source_column TEXT,
        guideline_reference TEXT,
        FOREIGN KEY (condition_id) REFERENCES conditions(condition_id)
    )
""")

# Reasons table removed (no warrants in strict PDF-faithful version)
print("ℹ️  Reasons table skipped (no warrants in strict PDF-faithful version)")

conn.commit()

# Verify schema
print("\n✅ Schema created successfully!")
print("\n📊 Table structure:")
for table in ["conditions", "arguments", "recommendations", "tests", "ultrasound"]:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"  {table}: {columns}")

conn.close()
print("\n🔒 Database connection closed.")