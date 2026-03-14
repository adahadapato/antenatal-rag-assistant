# minimal_example.py
from pipeline_agent_sqlite import query_guidelines
#from pipeline_agent_json import query_guidelines
import json

# Define patient
patient = {
  "age": 42,
  "gestational_age_weeks":28 ,
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
    query="What care is needed for anaemia?",
    patient_context=patient
)

# Output as JSON (for team leader)
print(json.dumps(result, indent=2))