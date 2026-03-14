# verify_unicode.py
import json
import sqlite3
import Initialize

print("=" * 80)
print("UNICODE VERIFICATION CHECK")
print("=" * 80)

# Check 1: Verify SQLite database content
print("\n1. Checking SQLite database...")
conn = sqlite3.connect(Initialize.augmented_db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT claim FROM arguments WHERE claim LIKE '%–%' LIMIT 3")
rows = cursor.fetchall()

if rows:
    print(f"   ✅ Found {len(rows)} claims with en-dash character (–)")
    for row in rows:
        claim = row['claim']
        if '–' in claim:
            print(f"      • {claim[:80]}...")
        elif '\\u2013' in claim:
            print(f"      ❌ Found escape sequence: {claim[:80]}...")
else:
    print("   ⚠️  No claims with en-dash found")

conn.close()

# Check 2: Verify JSON output
print("\n2. Checking JSON output...")
from pipeline_agent_sqlite import query_guidelines

result = query_guidelines(query="What care for BMI?", patient_context={"bmi": 45.0})

# Check if clarify section exists
if result.get('sections', {}).get('clarify'):
    print(f"   ✅ Clarify section present with {len(result['sections']['clarify'])} items")
    
    # Check first clarify item for en-dash
    first_clarify = result['sections']['clarify'][0]
    if '–' in first_clarify.get('question', ''):
        print(f"   ✅ En-dash character (–) found in clarify questions")
    elif '\\u2013' in str(first_clarify):
        print(f"   ❌ Escape sequence (\\u2013) found instead of en-dash")
else:
    print("   ⚠️  Clarify section missing")

# Check 3: Verify guideline_reference field
print("\n3. Checking guideline_reference field...")
if result.get('sections', {}).get('actions'):
    first_action = result['sections']['actions'][0]
    if 'guideline_reference' in first_action:
        print(f"   ✅ guideline_reference present: {first_action['guideline_reference'][:50]}...")
    else:
        print(f"   ❌ guideline_reference missing from actions")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)