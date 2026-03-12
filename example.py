# minimal_example.py
from pipeline_agent import query_guidelines
import json

# Define patient
patient = {
  "age": 42,
  "gestational_age_weeks": null,
  "bmi": 45.0,
  "parity": 2,
  "medical_history": [
    "previous history of DVT"
  ],
  "obstetric_history": [],
  "current_medications": []
}

# Query guidelines
result = query_guidelines(
    query="What antenatal care is needed?",
    patient_context=patient
)

# Output as JSON (for team leader)
print(json.dumps(result, indent=2))