# ask.py - Optimized for Argumentation Theory Database with JSON API Support
# =============================================================================
# PURPOSE:
#   Hybrid retrieval system for antenatal care guidelines.
#   - First attempts structured lookup in argumentation database
#   - Falls back to vector RAG (Chroma + Ollama) for general queries
#   - Includes source tracing, fuzzy matching, and deduplication
#   - Outputs structured JSON for UI/API integration
#
# ARCHITECTURE:
#   1. Load argumentation JSON database (Toulmin-style schema)
#   2. Load Chroma vector store for unstructured PDF content
#   3. For each user question:
#      a. Try structured DB lookup with fuzzy matching
#      b. If found: format and return structured answer (text + JSON)
#      c. If not found: retrieve from Chroma, expand via LinkRetrieverAgent,
#         and generate answer via local Ollama LLM
#   4. Always display source references with deduplication
#
# USAGE:
#   python ask.py                    # CLI mode (formatted text)
#   python ask.py --json            # JSON mode (structured output)
#   python ask.py --json --quiet    # JSON only (no text output)
#
#   Type questions like:
#     - "What care is needed for anaemia?"
#     - "When should I scan for twins?"
#     - "Why is aspirin given at 12 weeks?"
#   Type 'exit' or 'quit' to stop
# =============================================================================

# Import your path/config file.
import Initialize
from pathlib import Path
import sys
import argparse
import json as json_module  # Alias to avoid conflict with variable

# Import the link-following agent for linked PDFs.
from link_agent import LinkRetrieverAgent

# Import embedding model wrapper for Chroma retrieval.
from langchain_huggingface import HuggingFaceEmbeddings

# Import Chroma vector database.
from langchain_community.vectorstores import Chroma

# Import local Ollama chat model.
from langchain_community.chat_models import ChatOllama

# Import prompt template for the LLM.
from langchain_core.prompts import ChatPromptTemplate

# Import the built-in JSON module so we can read the argumentation database.
import re  # For text processing and citation cleanup

# Optional: fuzzy matching for better condition detection
try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️  fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")


# =============================================================================
# LOAD THE ARGUMENTATION DATABASE
# =============================================================================

# Store the path to the argumentation JSON file.
ARG_DB_PATH = Initialize.augmented_db_json

# Open the JSON file safely in read mode.
with open(ARG_DB_PATH, "r", encoding="utf-8") as f:
    # Load the JSON into Python as a list of dictionaries.
    argument_data = json_module.load(f)

# FIX: Check if data is nested inside "arguments" key (handles wrapper object)
if isinstance(argument_data, dict) and "arguments" in argument_data:
    argument_data = argument_data["arguments"]

# Pre-build a lookup dictionary for faster condition matching
# Key: lowercase condition name, Value: full condition record
CONDITION_LOOKUP = {item["condition_name"].lower(): item for item in argument_data}
CONDITION_NAMES = list(CONDITION_LOOKUP.keys())


# =============================================================================
# HELPER: FIND A CONDITION IN THE ARGUMENT DB
# =============================================================================

def find_condition_in_question(question: str):
    """
    Try to identify whether the user's question mentions
    one of the known conditions in the argumentation database.
    
    Uses a three-tier approach:
    1. Exact substring match (fast)
    2. Fuzzy matching if available (flexible)
    3. Synonym mapping for common medical terms
    
    Returns the matching condition record if found, else None.
    """
    q_lower = question.lower()
    
    # Tier 1: Try exact substring match first (fast path)
    for condition_name in CONDITION_NAMES:
        if condition_name in q_lower:
            return CONDITION_LOOKUP[condition_name]
    
    # Tier 2: Try fuzzy matching if available (more flexible)
    if FUZZY_AVAILABLE:
        # Extract potential condition keywords from question
        # score_cutoff=60 means >=60% similarity required
        match = process.extractOne(q_lower, CONDITION_NAMES, score_cutoff=60)
        if match:
            matched_name = match[0]
            return CONDITION_LOOKUP[matched_name]
    
    # Tier 3: Try synonym mapping for common medical terms
    # Maps colloquial/abbreviated terms to formal condition names
    synonym_map = {
        "high blood pressure": "chronic hypertension",
        "hypertension": "chronic hypertension",
        "htn": "chronic hypertension",
        "low iron": "anaemia",
        "low hb": "anaemia",
        "gestational diabetes": "gestational diabetes",
        "gdm": "gestational diabetes",
        "reduced movements": "reduced fetal movements",
        "rfm": "reduced fetal movements",
        "small baby": "sga baby",
        "large baby": "large baby",
        "lga": "large baby",
        "fgr": "fgr baby",
        "growth restriction": "fgr baby",
        "placenta previa": "placenta praevia / low lying",
        "previa": "placenta praevia / low lying",
        "twin": "twins",
        "twins dcda": "twins (dcda)",
        "twins mcda": "twins (mcda)",
        "diabetes type 1": "type 1 and 2 diabetes (pre-pregnant)",
        "diabetes type 2": "type 1 and 2 diabetes (pre-pregnant)",
        "pre-existing diabetes": "type 1 and 2 diabetes (pre-pregnant)",
    }
    
    for synonym, condition_key in synonym_map.items():
        if synonym in q_lower and condition_key in CONDITION_LOOKUP:
            return CONDITION_LOOKUP[condition_key]
    
    return None


# =============================================================================
# HELPER: DETECT WHAT TYPE OF INFORMATION IS ASKED FOR
# =============================================================================

def detect_question_type(question: str):
    """
    Decide what the user is asking for:
    recommendations, tests, ultrasound, reasons, or full summary.
    
    Uses weighted keyword matching for better accuracy than simple
    substring checks. Returns the category with highest keyword match score.
    """
    q_lower = question.lower()
    
    # Define keyword weights for each category
    # Keywords are medical/clinical terms relevant to each information type
    categories = {
        "tests": {
            "keywords": ["test", "tests", "examination", "examinations", "check", "checks", 
                        "ogtt", "blood", "bp", "urine", "fbc", "ferritin", "hba1c", "tft", 
                        "g&s", "group and save", "antibody", "titre", "rpr", "vdrl"],
            "weight": 1
        },
        "ultrasound": {
            "keywords": ["scan", "scans", "ultrasound", "uss", "echo", "doppler", "mca psv", 
                        "uterine artery", "growth scan", "anomaly scan", "dating scan", 
                        "cervical length", "placental location"],
            "weight": 1
        },
        "reasons": {
            "keywords": ["why", "reason", "risk", "rationale", "justify", "because", 
                        "purpose", "benefit", "indication", "prophylaxis"],
            "weight": 1
        },
        "recommendations": {
            "keywords": ["follow", "follow-up", "management", "recommend", "recommended", 
                        "care", "review", "referral", "counselling", "counseling", 
                        "what should be done", "plan", "treatment", "intervention", 
                        "appoint", "appointment", "i ol", "induction", "delivery"],
            "weight": 1
        }
    }
    
    # Calculate scores for each category based on keyword matches
    scores = {}
    for category, config in categories.items():
        score = sum(config["weight"] for kw in config["keywords"] if kw in q_lower)
        if score > 0:
            scores[category] = score
    
    # Return the category with highest score, or default to summary
    if scores:
        return max(scores, key=scores.get)
    
    return "summary"


# =============================================================================
# HELPER: CLASSIFY ARGUMENT CLAIMS INTO CATEGORIES
# =============================================================================

def classify_claim(claim: str) -> str:
    """
    Classify a claim into: recommendation, test, ultrasound, or reason.
    Used to organize nested arguments for display in the formatted answer.
    
    Classification is based on keyword presence in the claim text.
    """
    claim_lower = claim.lower()
    
    if any(kw in claim_lower for kw in ["scan", "ultrasound", "uss", "echo", "doppler", "mca", "cervical length"]):
        return "ultrasound"
    elif any(kw in claim_lower for kw in ["test", "blood", "ogtt", "fbc", "ferritin", "hba1c", "tft", "g&s", "antibody", "rpr", "vdrl", "urine", "bp"]):
        return "tests"
    elif any(kw in claim_lower for kw in ["why", "because", "risk", "prophylaxis", "prevent", "reduce"]):
        return "reasons"
    else:
        return "recommendations"


# =============================================================================
# HELPER: FORMAT THE ARGUMENT DB ANSWER FOR THE USER (TEXT)
# =============================================================================

def format_argument_answer(item, question_type: str):
    """
    Build a clinical decision support answer from the structured argumentation database.
    Output format matches clinical workflow: Guidelines → Actions → Tests → Ultrasound → Follow Up → Clarify → Decision Points → Plan
    Returns formatted string for CLI display.
    """
    lines = []
    
    # Header
    condition_name = item.get("condition_name", "Unknown Condition")
    criteria = item.get("criteria", "")
    source_page = item.get("source_page", "N/A")
    source_ref = item.get("source_reference_file", "")
    
    lines.append(f"🩺 Condition: {condition_name}")
    if criteria:
        lines.append(f"📋 Criteria: {criteria}")
    lines.append(f"📄 Source: Page {source_page}" + (f" • {source_ref}" if source_ref else ""))
    lines.append("")
    
    # Extract and categorize arguments
    actions = []
    tests = []
    ultrasounds = []
    follow_up_items = []
    clarify_items = []
    decision_points = []
    
    for arg in item.get("arguments", []):
        claim = arg.get("claim", "")
        timing = arg.get("timing", "")
        category = arg.get("category", "recommendation")
        source_column = arg.get("source_column", "")
        
        full_entry = f"{claim}" + (f" ({timing})" if timing and timing != "As per guideline" else "")
        
        # Categorize based on JSON category field
        if category == "ultrasound" or "scan" in claim.lower() or "doppler" in claim.lower():
            ultrasounds.append(full_entry)
        elif category == "test" or any(kw in claim.lower() for kw in ["blood", "fbc", "ogtt", "g&s", "tft", "urine", "bp", "ferritin", "hba1c", "antibody"]):
            tests.append(full_entry)
        elif any(kw in claim.lower() for kw in ["follow", "review", "next visit", "escalate", "refer"]):
            follow_up_items.append(full_entry)
        elif any(kw in claim.lower() for kw in ["if", "unless", "when", "consider", "assess"]):
            # These are potential decision points or clarifying questions
            if "?" in claim or any(kw in claim.lower() for kw in ["history", "previous", "known", "detected"]):
                clarify_items.append(claim)
            else:
                decision_points.append({"condition": claim.split("if")[0].strip() if "if" in claim.lower() else claim, "action": claim})
        else:
            actions.append(full_entry)
    
    # ========== GUIDELINES SECTION ==========
    lines.append("📚 Guidelines:")
    if source_ref and source_ref != "ANTENATAL CARE SCHEDULE.pdf":
        lines.append(f"  • {source_ref}")
    lines.append(f"  • ANTENATAL CARE SCHEDULE  (Page {source_page})")
    lines.append("")
    
    # ========== ACTIONS SECTION ==========
    if actions:
        lines.append("✅ Actions:")
        for a in actions:
            lines.append(f"  • {a}")
        lines.append("")
    
    # ========== TESTS SECTION ==========
    if tests:
        lines.append("🧪 Tests:")
        for t in tests:
            lines.append(f"  • {t}")
        lines.append("")
    
    # ========== ULTRASOUND SECTION ==========
    if ultrasounds:
        lines.append("📡 Ultrasound:")
        for u in ultrasounds:
            lines.append(f"  • {u}")
        lines.append("")
    
    # ========== FOLLOW UP SECTION ==========
    if follow_up_items:
        lines.append("🔄 Follow Up:")
        for f in follow_up_items:
            lines.append(f"  • {f}")
        lines.append("")
    else:
        # Default follow-up based on timing in actions
        default_followup = [a for a in actions if any(kw in a.lower() for kw in ["weeks", "booking", "visit"])]
        if default_followup:
            lines.append("🔄 Follow Up:")
            for f in default_followup[:3]:  # Show top 3 timing-related items
                lines.append(f"  • {f}")
            lines.append("")
    
    # ========== CLARIFY SECTION ==========
    if clarify_items or criteria:
        lines.append("❓ Clarify:")
        if criteria:
            lines.append(f"  • Does patient meet criteria: {criteria}?")
        for c in clarify_items:
            lines.append(f"  • {c}?")
        lines.append("")
    
    # ========== DECISION POINTS SECTION (Collapsible-style) ==========
    # Extract IF-THEN patterns from claims
    if_condition_patterns = [arg for arg in item.get("arguments", []) if "if" in arg.get("claim", "").lower() or "unless" in arg.get("claim", "").lower()]
    
    if if_condition_patterns:
        lines.append("🔀 Decision Points ▼")  # ▼ indicates collapsible
        for dp in if_condition_patterns:
            claim = dp.get("claim", "")
            timing = dp.get("timing", "")
            # Simple parsing: split on "if" or "unless"
            if " if " in claim.lower():
                parts = claim.lower().split(" if ", 1)
                condition = parts[1].strip() if len(parts) > 1 else "condition"
                action = parts[0].strip()
            elif " unless " in claim.lower():
                parts = claim.lower().split(" unless ", 1)
                condition = f"NOT ({parts[1].strip()})" if len(parts) > 1 else "condition"
                action = parts[0].strip()
            else:
                condition = "See criteria"
                action = claim
            
            lines.append(f"    ├─ IF {condition.capitalize()}")
            lines.append(f"    │  → {action.capitalize()}" + (f" ({timing})" if timing else ""))
        lines.append("")
    
    # ========== HAVE PLAN SECTION ==========
    lines.append("📅 Have Plan:")
    # Extract next visit timing from arguments
    next_visits = [arg.get("timing", "") for arg in item.get("arguments", []) if arg.get("timing") and arg.get("timing") not in ["As per guideline", "N/A", ""]]
    if next_visits:
        # Show earliest upcoming timing
        upcoming = [t for t in next_visits if any(kw in t.lower() for kw in ["weeks", "booking", "next"])]
        if upcoming:
            lines.append(f"  • Next review: {upcoming[0]}")
        else:
            lines.append(f"  • Follow schedule per guidelines")
    else:
        lines.append("  • Routine antenatal schedule unless otherwise indicated")
    
    # Add exceptions/rebuttals if any
    rebuttals = [arg.get("rebuttal") for arg in item.get("arguments", []) if arg.get("rebuttal") and arg.get("rebuttal") != "None specified"]
    if rebuttals:
        lines.append("")
        lines.append("⚠️ Exceptions:")
        for r in rebuttals:
            lines.append(f"  • {r}")
    
    return "\n".join(lines)


# =============================================================================
# HELPER: GENERATE STRUCTURED JSON RESPONSE FOR UI/API
# =============================================================================

def generate_json_response(item, question: str, question_type: str, source: str = "structured_db"):
    """
    Generate a structured JSON response suitable for UI/API consumption.
    
    Returns a dictionary with:
    - metadata: query info, timestamp, source
    - condition: condition details
    - sections: Guidelines, Actions, Tests, Ultrasound, FollowUp, Clarify, DecisionPoints, Plan
    - raw_data: original arguments for debugging
    """
    # Extract condition info
    condition_name = item.get("condition_name", "Unknown Condition")
    criteria = item.get("criteria", "")
    source_page = item.get("source_page", "N/A")
    source_ref = item.get("source_reference_file", "")
    
    # Extract and categorize arguments
    actions = []
    tests = []
    ultrasounds = []
    follow_up_items = []
    clarify_items = []
    decision_points = []
    
    for arg in item.get("arguments", []):
        claim = arg.get("claim", "")
        timing = arg.get("timing", "")
        category = arg.get("category", "recommendation")
        source_column = arg.get("source_column", "")
        
        entry = {
            "claim": claim,
            "timing": timing if timing and timing != "As per guideline" else None,
            "category": category,
            "source_column": source_column
        }
        
        # Categorize based on JSON category field
        if category == "ultrasound" or "scan" in claim.lower() or "doppler" in claim.lower():
            ultrasounds.append(entry)
        elif category == "test" or any(kw in claim.lower() for kw in ["blood", "fbc", "ogtt", "g&s", "tft", "urine", "bp", "ferritin", "hba1c", "antibody"]):
            tests.append(entry)
        elif any(kw in claim.lower() for kw in ["follow", "review", "next visit", "escalate", "refer"]):
            follow_up_items.append(entry)
        elif any(kw in claim.lower() for kw in ["if", "unless", "when", "consider", "assess"]):
            if "?" in claim or any(kw in claim.lower() for kw in ["history", "previous", "known", "detected"]):
                clarify_items.append(claim)
            else:
                # Parse IF-THEN for decision points
                if " if " in claim.lower():
                    parts = claim.lower().split(" if ", 1)
                    condition = parts[1].strip() if len(parts) > 1 else "condition"
                    action = parts[0].strip()
                elif " unless " in claim.lower():
                    parts = claim.lower().split(" unless ", 1)
                    condition = f"NOT ({parts[1].strip()})" if len(parts) > 1 else "condition"
                    action = parts[0].strip()
                else:
                    condition = "See criteria"
                    action = claim
                
                decision_points.append({
                    "condition": condition,
                    "action": action,
                    "timing": timing if timing and timing != "As per guideline" else None,
                    "original_claim": claim
                })
        else:
            actions.append(entry)
    
    # Build decision points from IF-THEN patterns
    if_condition_patterns = [arg for arg in item.get("arguments", []) if "if" in arg.get("claim", "").lower() or "unless" in arg.get("claim", "").lower()]
    for dp in if_condition_patterns:
        claim = dp.get("claim", "")
        timing = dp.get("timing", "")
        if " if " in claim.lower():
            parts = claim.lower().split(" if ", 1)
            condition = parts[1].strip() if len(parts) > 1 else "condition"
            action = parts[0].strip()
        elif " unless " in claim.lower():
            parts = claim.lower().split(" unless ", 1)
            condition = f"NOT ({parts[1].strip()})" if len(parts) > 1 else "condition"
            action = parts[0].strip()
        else:
            continue  # Skip if no clear IF-THEN
        
        # Avoid duplicates
        if not any(d["original_claim"] == claim for d in decision_points):
            decision_points.append({
                "condition": condition,
                "action": action,
                "timing": timing if timing and timing != "As per guideline" else None,
                "original_claim": claim
            })
    
    # Determine next review timing
    next_visits = [arg.get("timing", "") for arg in item.get("arguments", []) if arg.get("timing") and arg.get("timing") not in ["As per guideline", "N/A", ""]]
    upcoming = [t for t in next_visits if any(kw in t.lower() for kw in ["weeks", "booking", "next"])]
    next_review = upcoming[0] if upcoming else ("Follow schedule per guidelines" if next_visits else "Routine antenatal schedule unless otherwise indicated")
    
    # Collect exceptions
    rebuttals = [arg.get("rebuttal") for arg in item.get("arguments", []) if arg.get("rebuttal") and arg.get("rebuttal") != "None specified"]
    
    # Build response
    response = {
        "metadata": {
            "query": question,
            "question_type": question_type,
            "source": source,
            "timestamp": None,  # Will be set at runtime
            "condition_matched": condition_name
        },
        "condition": {
            "id": item.get("condition_id"),
            "name": condition_name,
            "criteria": criteria,
            "source": {
                "page": source_page,
                "section": item.get("source_section"),
                "reference_file": source_ref if source_ref != "ANTENATAL CARE SCHEDULE.pdf" else None
            }
        },
        "sections": {
            "guidelines": [
                {"text": source_ref} if source_ref and source_ref != "ANTENATAL CARE SCHEDULE.pdf" else None,
                {"text": f"ANTENATAL CARE SCHEDULE CHEAT SHEET (Page {source_page})"}
            ],
            "actions": actions if actions else None,
            "tests": tests if tests else None,
            "ultrasound": ultrasounds if ultrasounds else None,
            "follow_up": follow_up_items if follow_up_items else (default_followup[:3] if (default_followup := [a for a in actions if any(kw in a.get("claim", "").lower() for kw in ["weeks", "booking", "visit"])]) else None),
            "clarify": [f"Does patient meet criteria: {criteria}?"] + clarify_items if (clarify_items or criteria) else None,
            "decision_points": decision_points if decision_points else None,
            "plan": {
                "next_review": next_review,
                "exceptions": rebuttals if rebuttals else None
            }
        },
        "raw_data": {
            "arguments": item.get("arguments", [])
        }
    }
    
    # Clean up None values in guidelines
    response["sections"]["guidelines"] = [g for g in response["sections"]["guidelines"] if g is not None]
    
    return response

# =============================================================================
# GET JSON OBJECT OF PATIENTS CONTEXT FROM ALLAN MODULE
# =============================================================================
def parse_patient_context(patient_input: str) -> dict:
    """
    Parse patient JSON from string or file path.
    Returns normalized patient context dict.
    """
    import os
    import json as json_module
    
    # Check if input is a file path
    if os.path.exists(patient_input):
        with open(patient_input, 'r', encoding='utf-8') as f:
            return json_module.load(f)
    
    # Otherwise treat as JSON string
    try:
        return json_module.loads(patient_input)
    except json_module.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return {}
    
def get_triggered_conditions(patient: dict) -> list[str]:
    """
    Determine which conditions are triggered by patient parameters.
    Returns list of condition_name values that should be auto-included.
    """
    triggered = []
    
    # Age-based triggers
    if patient.get("age", 0) >= 40:
        triggered.append("advanced maternal age-> 40")
    
    # BMI-based triggers
    bmi = patient.get("bmi")
    if bmi is not None:
        if 35 <= bmi < 40:
            triggered.append("bmi 35-39.9")
        elif bmi >= 40:
            triggered.append("bmi > 40")
        elif bmi < 18.5:
            triggered.append("low bmi < 18.5")
    
    # Medical history triggers
    medical_history = " ".join(patient.get("medical_history", [])).lower()
    if "dvt" in medical_history or "thrombosis" in medical_history or "clot" in medical_history:
        # VTE risk - boost BMI recommendations that mention LMWH
        triggered.append("bmi > 40")  # Ensure BMI pathway is included for VTE assessment
    if "diabetes" in medical_history or "type 1" in medical_history or "type 2" in medical_history:
        triggered.append("type 1 and 2 diabetes - pre-pregnant diabetes")
    if "hypertension" in medical_history or "htn" in medical_history:
        triggered.append("chronic hypertension")
    if "epilepsy" in medical_history or "seizure" in medical_history:
        triggered.append("epilepsy")
    if "sickle" in medical_history or "haemoglobinopathy" in medical_history:
        triggered.append("sickle cell disease")
    
    # Obstetric history triggers
    obstetric_history = " ".join(patient.get("obstetric_history", [])).lower()
    if "pre-eclampsia" in obstetric_history or "pet" in obstetric_history:
        triggered.append("previous pet")
    if "caesarean" in obstetric_history or "c-section" in obstetric_history:
        triggered.append("previous caesarean section")
    if "twins" in obstetric_history or "multiple" in obstetric_history:
        triggered.append("twins")  # Will need user to specify DCDA/MCDA
    
    return triggered

# =============================================================================
# MAIN PROGRAM FLOW
# =============================================================================

def main():
    """
    Main entry point for the antenatal care Q&A system.
    
    Implements hybrid retrieval:
    1. Try structured argumentation database first
    2. Fall back to Chroma + Ollama RAG if no structured match
    3. Always deduplicate source references for clean output
    4. Support JSON output for UI/API integration
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Antenatal Care Guidelines Q&A System")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress text output (JSON mode only)")
    args = parser.parse_args()
    
    # Get the Chroma DB path from Initialize.py.
    chroma_path = Initialize.vector_db_path

    # Load the same embedding model used when the vector DB was created.
    # all-mpnet-base-v2 is a strong general-purpose embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Load the existing Chroma vector store from disk.
    vectordb = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    # Create the link agent for optional linked-document ingestion.
    # Use pathlib for cross-platform path handling (Windows/Linux/Mac)
    base_dir = Path(__file__).parent
    agent = LinkRetrieverAgent(
        linked_docs_dir=base_dir / "linked_docs",
        state_dir=base_dir / ".agent_state",
        max_new_docs_per_question=2
    )

    # Create the local Ollama model.
    # gemma3:4b is a compact model suitable for local deployment
    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0  # Deterministic output for clinical consistency
    )

    # Create the prompt used for fallback RAG answers.
    # System prompt enforces strict grounding in provided context
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a medical assistant for antenatal care guidelines. "
            "Answer ONLY using the provided context. "
            "Answer in ENGLISH. Do not summarise unrelated parts of the document. "
            "If the answer is not in the context, say: "
            "'I don't know based on the document.' "
            "Always include source page references when available."
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )
    ])

    # Print startup message with system status (unless quiet mode)
    if not args.quiet:
        print(f"✅ Loaded Chroma DB from: {chroma_path}")
        print(f"✅ Loaded {len(argument_data)} conditions from argumentation database")
        print(f"✅ Fuzzy matching: {'Enabled' if FUZZY_AVAILABLE else 'Disabled (install fuzzywuzzy for better matching)'}")
        print(f"✅ Output mode: {'JSON' if args.json else 'Formatted Text'}")
        print("\n💡 Tip: Ask about specific conditions like 'anaemia', 'gestational diabetes', or 'twins'")
        print("Type a question (or type 'exit')\n")

    # Start an infinite loop so the user can ask multiple questions.
    while True:
        try:
            # Read the user's question from the terminal.
            question = input("❓ Question: ").strip()

            # Stop the program if the user types exit or quit.
            if question.lower() in {"exit", "quit", "q"}:
                if not args.quiet:
                    print("👋 Goodbye!")
                break

            if not question:
                continue

            # =================================================================
            # STEP 1: TRY STRUCTURED ARGUMENT DATABASE FIRST
            # =================================================================

            # Try to find a matching condition inside the user's question.
            matched_item = find_condition_in_question(question)

            # If a condition was found, answer directly from the structured DB.
            if matched_item is not None:
                # Detect what part of the condition the user is asking about.
                question_type = detect_question_type(question)

                # Generate both text and JSON responses
                answer_text = format_argument_answer(matched_item, question_type)
                json_response = generate_json_response(matched_item, question, question_type, source="structured_db")
                json_response["metadata"]["timestamp"] = __import__("datetime").datetime.now().isoformat()
                
                # Output based on mode
                if args.json:
                    # Output JSON (to stdout for piping/API)
                    print(json_module.dumps(json_response, indent=2, ensure_ascii=False))
                if not args.quiet and not args.json:
                    # Output formatted text
                    print("\n" + "═" * 70)
                    print("✅ Answer from Argumentation DB:")
                    print("═" * 70)
                    print(answer_text)
                    print("═" * 70 + "\n")

                # Go to the next question without using RAG.
                continue

            # =================================================================
            # STEP 2: FALL BACK TO CHROMA + OLLAMA (RAG)
            # =================================================================

            # Use query boosting for better retrieval on specific topics
            boosted_query = question
            boost_terms = []
            
            if any(term in question.lower() for term in ["preconception", "before pregnancy", "planning pregnancy"]):
                boost_terms.append("preconception folic acid")
            if any(term in question.lower() for term in ["aspirin", "prophylaxis", "pre-eclampsia"]):
                boost_terms.append("aspirin 150mg pre-eclampsia prophylaxis")
            if any(term in question.lower() for term in ["growth", "fgr", "sga", "small baby"]):
                boost_terms.append("fetal growth restriction scan monitoring")
            
            if boost_terms:
                boosted_query = " ".join(boost_terms) + " " + question

            # Retrieve similar chunks from Chroma.
            # k=4 returns top 4 most relevant document chunks
            docs = vectordb.similarity_search(boosted_query, k=4)

            # Let the link agent inspect those chunks for referenced PDFs.
            # This enables dynamic ingestion of linked guideline documents.
            actions = agent.expand_and_ingest(docs, vectordb)

            # Print any actions performed by the agent (only success messages, unless quiet).
            if not args.quiet:
                for a in actions:
                    if a.startswith("✅"):
                        print(f"🔗 {a}")

            # If any new linked docs were ingested, run retrieval again.
            # This ensures newly-ingested content is considered for the answer.
            if any(a.startswith("✅") for a in actions):
                docs = vectordb.similarity_search(boosted_query, k=4)

            # Build one context string from the retrieved documents.
            # Include page numbers and source filenames for traceability.
            context_parts = []
            for d in docs:
                page = d.metadata.get('page', 'N/A')
                source = d.metadata.get('source', '')
                part = f"[Page {page}]" + (f" ({source})" if source else "") + f"\n{d.page_content}"
                context_parts.append(part)
            context = "\n\n---\n\n".join(context_parts)

            # Build the chain by combining the prompt and the LLM.
            chain = prompt | llm

            # Invoke the chain with the question and retrieved context.
            answer = chain.invoke({
                "question": question,
                "context": context
            })

            # Prepare RAG JSON response
            rag_json_response = {
                "metadata": {
                    "query": question,
                    "source": "vector_rag",
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "condition_matched": None,
                    "retrieved_pages": list(set([d.metadata.get('page', 'N/A') for d in docs]))
                },
                "answer": {
                    "text": answer.content,
                    "sources": []
                },
                "raw_context": [
                    {
                        "page": d.metadata.get('page', 'N/A'),
                        "source": d.metadata.get('source', ''),
                        "content": d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
                    }
                    for d in docs
                ]
            }
            
            # Deduplicate sources for display
            if "don't know" not in answer.content.lower() and docs:
                seen_sources = set()
                for d in docs:
                    page = d.metadata.get('page', 'N/A')
                    source = d.metadata.get('source', d.metadata.get('filename', 'Unknown'))
                    key = (page, source)
                    if key not in seen_sources:
                        seen_sources.add(key)
                        rag_json_response["answer"]["sources"].append({
                            "page": page,
                            "filename": Path(source).name,
                            "full_path": str(source)
                        })

            # Output based on mode
            if args.json:
                # Output JSON (to stdout for piping/API)
                print(json_module.dumps(rag_json_response, indent=2, ensure_ascii=False))
            if not args.quiet and not args.json:
                # Output formatted text
                print("\n" + "─" * 70)
                print("✅ Answer from Chroma RAG:")
                print("─" * 70)
                print(answer.content)
                
                # Only print sources if the model gave a grounded answer.
                if "don't know" not in answer.content.lower() and rag_json_response["answer"]["sources"]:
                    print("\n📌 Sources referenced:")
                    for i, src in enumerate(rag_json_response["answer"]["sources"], start=1):
                        print(f"  {i}. Page {src['page']} • {src['filename']}")
                    
                    # Optional: Show consolidation info if chunks were deduplicated
                    if len(rag_json_response["answer"]["sources"]) < len(docs):
                        print(f"  ℹ️  Consolidated {len(docs)} chunks into {len(rag_json_response['answer']['sources'])} unique pages")
                        
                elif "don't know" in answer.content.lower():
                    print("\n💡 Tip: Try asking about a specific condition like 'anaemia' or 'gestational diabetes'")

                print("─" * 70 + "\n")

        except KeyboardInterrupt:
            if not args.quiet:
                print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            # Graceful error handling to keep the CLI responsive
            error_response = {
                "error": str(e),
                "query": question if 'question' in locals() else None,
                "timestamp": __import__("datetime").datetime.now().isoformat()
            }
            if args.json:
                print(json_module.dumps({"error": error_response}, indent=2))
            if not args.quiet and not args.json:
                print(f"\n❌ Error: {e}")
                print("💡 Continuing. Try rephrasing your question.\n")


# Run the main function only when this file is executed directly.
if __name__ == "__main__":
    main()