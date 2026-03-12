# populate_augment_db.py - Import argumentation data into SQLite
import sqlite3
import json
import Initialize
from pathlib import Path

db_path = Initialize.augmented_db_path
json_path = Initialize.augmented_db_json

# Verify files exist
if not Path(db_path).exists():
    print(f"❌ Database not found at {db_path}")
    print("💡 Run setup_db_schema.py first!")
    exit(1)

if not Path(json_path).exists():
    print(f"❌ JSON file not found at {json_path}")
    exit(1)

# Load argumentation data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle wrapper object if present (from our JSON structure)
if isinstance(data, dict) and "arguments" in data:
    conditions = data["arguments"]
else:
    conditions = data

print(f"📊 Loaded {len(conditions)} conditions from JSON")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")

# Clear existing data (in correct order - NO reasons table)
print("🗑️  Clearing existing data...")
cursor.execute("DELETE FROM ultrasound")
cursor.execute("DELETE FROM tests")
cursor.execute("DELETE FROM recommendations")
cursor.execute("DELETE FROM arguments")
cursor.execute("DELETE FROM conditions")

# Insert conditions and related data
print("📝 Importing conditions...")
conditions_inserted = 0
arguments_inserted = 0

for condition in conditions:
    try:
        # Insert into conditions table
        cursor.execute("""
            INSERT INTO conditions (condition_id, condition_name, criteria, source_page, source_section, source_reference_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            condition.get("condition_id"),
            condition.get("condition_name"),
            condition.get("criteria"),
            condition.get("source_page"),
            condition.get("source_section"),
            condition.get("source_reference_file")
        ))
        
        conditions_inserted += 1
        
        # Get the condition_id for foreign key references
        cond_id = condition.get("condition_id")
        
        # Insert arguments (nested structure from our JSON)
        for arg in condition.get("arguments", []):
            claim = arg.get("claim", "")
            timing = arg.get("timing", "")
            category = arg.get("category", "recommendation")
            source_column = arg.get("source_column", "Antenatal Visits")
            arg_id = arg.get("arg_id", "")
            
            # Insert into arguments table (keeps original structure)
            cursor.execute("""
                INSERT INTO arguments (condition_id, arg_id, claim, timing, category, source_column)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cond_id, arg_id, claim, timing, category, source_column))
            
            # Also categorize into specific tables for easier querying
            if category == "ultrasound":
                cursor.execute("""
                    INSERT INTO ultrasound (condition_id, claim, timing, category, source_column)
                    VALUES (?, ?, ?, ?, ?)
                """, (cond_id, claim, timing, category, source_column))
            elif category == "test":
                cursor.execute("""
                    INSERT INTO tests (condition_id, claim, timing, category, source_column)
                    VALUES (?, ?, ?, ?, ?)
                """, (cond_id, claim, timing, category, source_column))
            else:
                cursor.execute("""
                    INSERT INTO recommendations (condition_id, claim, timing, category, source_column)
                    VALUES (?, ?, ?, ?, ?)
                """, (cond_id, claim, timing, category, source_column))
            
            arguments_inserted += 1
        
    except Exception as e:
        print(f"⚠️  Error inserting condition {condition.get('condition_name')}: {e}")
        continue

conn.commit()

print(f"\n✅ Import complete!")
print(f"   Conditions inserted: {conditions_inserted}")
print(f"   Arguments inserted: {arguments_inserted}")

# Verify counts
print("\n📊 Database contents:")
for table in ["conditions", "arguments", "recommendations", "tests", "ultrasound"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"   {table}: {count} records")

conn.close()
print("\n🔒 Database connection closed.")