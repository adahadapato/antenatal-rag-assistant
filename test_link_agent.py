# test_link_agent.py - Updated to handle both response types
from pipeline_agent import query_guidelines
import json

# Test 1: Query that should match structured DB (no link agent needed)
print("=" * 80)
print("TEST 1: Structured DB Lookup (Anaemia)")
print("=" * 80)

result = query_guidelines(
    query="What care is needed for anaemia?",
    patient_context={"age": 30, "bmi": 25.0}
)

print(f"Source: {result['metadata']['source']}")

if result['metadata']['source'] == 'structured_db':
    # Structured response format
    print(f"Condition: {result['condition']['name']}")
    print(f"Criteria: {result['condition']['criteria']}")
    print(f"\nActions ({len(result['sections']['actions'])} items):")
    for action in result['sections']['actions'][:3]:  # Show first 3
        print(f"  • {action['claim']} ({action['timing']})")
        print(f"    Guideline: {action.get('guideline_reference', 'N/A')}")
else:
    # RAG response format
    print(f"Answer preview: {result['answer']['text'][:200]}...")
    print(f"Sources: {[s['filename'] for s in result['answer']['sources']]}")

print()

# Test 2: Query that might trigger RAG fallback (more specific/complex)
print("=" * 80)
print("TEST 2: RAG Fallback (Specific Guideline Query)")
print("=" * 80)

result = query_guidelines(
    query="What does the Hillingdon FGR guideline say about uterine artery doppler thresholds?",
    patient_context={"age": 35, "bmi": 38.0}
)

print(f"Source: {result['metadata']['source']}")

if result['metadata']['source'] == 'structured_db':
    print(f"Condition: {result['condition']['name']}")
    print(f"\nUltrasound recommendations:")
    for us in result['sections']['ultrasound']:
        print(f"  • {us['claim']}")
        print(f"    Guideline: {us.get('guideline_reference', 'N/A')}")
else:
    # RAG response format
    print(f"Answer preview: {result['answer']['text'][:200]}...")
    print(f"Sources referenced: {[s['filename'] for s in result['answer']['sources']]}")
    # Check if link agent was used
    if result['metadata'].get('retrieved_pages'):
        print(f"Pages retrieved: {result['metadata']['retrieved_pages']}")

print()

# Test 3: Verify guideline_reference field is present
print("=" * 80)
print("TEST 3: Verify guideline_reference Field")
print("=" * 80)

result = query_guidelines(
    query="What care for BMI > 40?",
    patient_context={"age": 28, "bmi": 42.0}
)

print(f"Condition: {result['condition']['name']}")
print(f"\nChecking guideline_reference in each section:")

for section_name in ['actions', 'tests', 'ultrasound', 'follow_up']:
    items = result['sections'].get(section_name, [])
    if items:
        print(f"\n{section_name.upper()} ({len(items)} items):")
        for item in items[:2]:  # Show first 2
            ref = item.get('guideline_reference', 'MISSING!')
            print(f"  • {item['claim'][:60]}...")
            print(f"    → Guideline: {ref}")

print("\n✅ All claims should have guideline_reference field populated")