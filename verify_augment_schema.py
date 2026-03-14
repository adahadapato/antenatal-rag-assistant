# verify_augment_schema.py
import sqlite3
import Initialize

conn = sqlite3.connect(Initialize.augmented_db_path)
cursor = conn.cursor()

# Check arguments table structure
cursor.execute("PRAGMA table_info(arguments)")
columns = [col[1] for col in cursor.fetchall()]
print(f"Arguments table columns: {columns}")

# Should include: ['id', 'condition_id', 'arg_id', 'claim', 'timing', 'category', 'source_column', 'guideline_reference']

# Sample query to verify guideline_reference is populated
cursor.execute("SELECT claim, guideline_reference FROM arguments LIMIT 3")
for row in cursor.fetchall():
    print(f"Claim: {row[0][:50]}...")
    print(f"Guideline: {row[1]}")
    print()

conn.close()