# pipeline_agent_sqlite.py - Updated to use SQLite database
# =============================================================================
# VERSION: 7.1
# CHANGE: Now loads from SQLite instead of JSON for consistency
# =============================================================================

import sqlite3
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

# Import path configuration
import Initialize

# Optional: fuzzy matching
try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


# =============================================================================
# GLOBAL STATE: GUIDELINE ENGINE (SINGLETON)
# =============================================================================

class GuidelineEngine:
    """
    Singleton engine for antenatal guideline queries.
    Loads data from SQLite database for consistency with project architecture.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GuidelineEngine._initialized:
            return
        
        # Load data from SQLite database
        self.argument_data = self._load_from_sqlite()
        
        # Build fast lookup dictionaries
        self.condition_lookup = {
            item["condition_name"].lower(): item 
            for item in self.argument_data
        }
        self.condition_names = list(self.condition_lookup.keys())
        
        # Lazy-loaded components
        self._vectordb = None
        self._embeddings = None
        self._llm = None
        
        GuidelineEngine._initialized = True
    
    def _load_from_sqlite(self) -> List[Dict]:
        """
        Load argumentation data from SQLite database.
        
        Returns:
            List[Dict]: List of condition records with arguments
        """
        db_path = Initialize.augmented_db_path
        
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                "Run setup_augment_db_schema.py and populate_augment_db.py first."
            )
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        try:
            # Get all conditions
            cursor.execute("""
                SELECT condition_id, condition_name, criteria, source_page, 
                       source_section, source_reference_file
                FROM conditions
            """)
            conditions = cursor.fetchall()
            
            result = []
            
            for condition in conditions:
                condition_dict = {
                    "condition_id": condition["condition_id"],
                    "condition_name": condition["condition_name"],
                    "criteria": condition["criteria"],
                    "source_page": condition["source_page"],
                    "source_section": condition["source_section"],
                    "source_reference_file": condition["source_reference_file"],
                    "arguments": []
                }
                
                # Get all arguments for this condition
                cursor.execute("""
                    SELECT arg_id, claim, timing, category, source_column, 
                           guideline_reference
                    FROM arguments
                    WHERE condition_id = ?
                """, (condition["condition_id"],))
                
                arguments = cursor.fetchall()
                
                for arg in arguments:
                    condition_dict["arguments"].append({
                        "arg_id": arg["arg_id"],
                        "claim": arg["claim"],
                        "timing": arg["timing"],
                        "category": arg["category"],
                        "source_column": arg["source_column"],
                        "guideline_reference": arg["guideline_reference"]
                    })
                
                result.append(condition_dict)
            
            return result
            
        finally:
            conn.close()
    
    @property
    def vectordb(self):
        """Lazy load vector database on first access."""
        if self._vectordb is None:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            self._vectordb = Chroma(
                persist_directory=Initialize.vector_db_path,
                embedding_function=self._embeddings
            )
        return self._vectordb
    
    @property
    def llm(self):
        """Lazy load Ollama LLM on first access."""
        if self._llm is None:
            from langchain_community.chat_models import ChatOllama
            self._llm = ChatOllama(model="gemma3:4b", temperature=0)
        return self._llm


# =============================================================================
# PUBLIC API: MAIN QUERY FUNCTION
# =============================================================================

def query_guidelines(
    query: str,
    patient_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main query function for antenatal care guidelines.
    
    Args:
        query (str): User's question about antenatal care
        patient_context (Dict, optional): Patient data for personalization
    
    Returns:
        Dict[str, Any]: Structured guideline response
    """
    engine = GuidelineEngine()
    
    try:
        # Get triggered conditions from patient context
        triggered = get_triggered_conditions(patient_context) if patient_context else []
        
        # Find matching condition(s) in structured database
        matched_item = find_condition_in_question(
            query, 
            triggered_conditions=triggered,
            condition_lookup=engine.condition_lookup,
            condition_names=engine.condition_names
        )
        
        if matched_item:
            # Return structured database response
            response = generate_json_response(
                item=matched_item,
                query=query,
                patient_context=patient_context,
                source="structured_db"
            )
        else:
            # Fallback to vector RAG
            response = query_vector_rag(
                query=query,
                patient_context=patient_context,
                engine=engine
            )
        
        return response
        
    except Exception as e:
        return create_error_response(error=str(e), query=query, patient_context=patient_context)


# =============================================================================
# HELPER FUNCTIONS (Same as before)
# =============================================================================

def get_triggered_conditions(patient: Optional[Dict[str, Any]]) -> List[str]:
    """Determine which conditions are triggered by patient parameters."""
    if not patient:
        return []
    
    triggered = []
    
    # Age-based triggers
    if patient.get("age", 0) >= 40:
        triggered.append("advanced maternal age-> 40")
    
    # BMI-based triggers (PDF-faithful categories)
    bmi = patient.get("bmi")
    if bmi is not None:
        if bmi < 18.5:
            triggered.append("low bmi < 18.5")
        elif 35 <= bmi < 40:
            triggered.append("bmi 35-39.9")
        elif bmi >= 40:
            triggered.append("bmi > 40")
    
    # Medical history triggers
    medical_history = " ".join(patient.get("medical_history", [])).lower()
    if "dvt" in medical_history or "thrombosis" in medical_history or "clot" in medical_history:
        triggered.append("bmi > 40")
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
        triggered.append("twins")
    
    return triggered


def find_condition_in_question(
    question: str,
    triggered_conditions: List[str] = None,
    condition_lookup: Dict = None,
    condition_names: List[str] = None
) -> Optional[Dict]:
    """Find matching condition(s) from question and triggered conditions."""
    if condition_lookup is None or condition_names is None:
        engine = GuidelineEngine()
        condition_lookup = engine.condition_lookup
        condition_names = engine.condition_names
    
    q_lower = question.lower()
    matched_items = []
    
    # Tier 1: Exact substring match
    for condition_name in condition_names:
        if condition_name in q_lower:
            matched_items.append(condition_lookup[condition_name])
    
    # Tier 2: Add triggered conditions
    if triggered_conditions:
        for trigger in triggered_conditions:
            for condition_name in condition_names:
                if trigger.lower() in condition_name and condition_lookup[condition_name] not in matched_items:
                    matched_items.append(condition_lookup[condition_name])
    
    # Tier 3: Fuzzy matching
    if FUZZY_AVAILABLE and not matched_items:
        match = process.extractOne(q_lower, condition_names, score_cutoff=60)
        if match:
            matched_items.append(condition_lookup[match[0]])
    
    # Tier 4: Synonym mapping
    if not matched_items:
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
            "twins dcda": "twins – dcda",
            "twins mcda": "twins – mcda",
            "diabetes type 1": "type 1 and 2 diabetes - pre-pregnant diabetes",
            "diabetes type 2": "type 1 and 2 diabetes - pre-pregnant diabetes",
            "pre-existing diabetes": "type 1 and 2 diabetes - pre-pregnant diabetes",
        }
        
        for synonym, condition_key in synonym_map.items():
            if synonym in q_lower:
                for cond_name in condition_names:
                    if condition_key.lower() in cond_name:
                        matched_items.append(condition_lookup[cond_name])
                        break
    
    if len(matched_items) == 1:
        return matched_items[0]
    elif len(matched_items) > 1:
        return merge_conditions(matched_items)
    
    return None


def merge_conditions(condition_list: List[Dict]) -> Dict:
    """Merge multiple condition records into a single response."""
    merged = {
        "condition_id": "MERGED-" + "-".join([c["condition_id"] for c in condition_list]),
        "condition_name": "Multiple Conditions: " + ", ".join([c["condition_name"] for c in condition_list]),
        "criteria": "Patient meets criteria for: " + "; ".join([c.get("criteria", "N/A") for c in condition_list]),
        "source_page": ", ".join([str(c.get("source_page", "N/A")) for c in condition_list]),
        "source_reference_file": ", ".join([c.get("source_reference_file", "N/A") for c in condition_list]),
        "arguments": []
    }
    
    seen_claims = set()
    for cond in condition_list:
        for arg in cond.get("arguments", []):
            claim_key = (arg.get("claim"), arg.get("timing"))
            if claim_key not in seen_claims:
                seen_claims.add(claim_key)
                merged["arguments"].append(arg)
    
    return merged


def categorize_bmi(bmi: float) -> str:
    """Categorize BMI based ONLY on PDF-defined thresholds."""
    if bmi is None:
        return "unknown"
    elif bmi < 18.5:
        return "low_bmi"
    elif bmi < 35:
        return "normal_range"
    elif bmi < 40:
        return "bmi_35_39.9"
    else:
        return "bmi_over_40"


def _claim_to_clarify_question(claim: str) -> Optional[str]:
    """Convert a conditional claim into a clarification question."""
    claim_lower = claim.lower()
    
    if " if " in claim_lower:
        parts = claim_lower.split(" if ", 1)
        condition = parts[1].strip().rstrip(').,;:')
        return f"Is {condition}?"
    
    if " unless " in claim_lower:
        parts = claim_lower.split(" unless ", 1)
        condition = parts[1].strip().rstrip(').,;:')
        return f"Is {condition} NOT present?"
    
    if any(kw in claim_lower for kw in ["known", "history of", "detected", "symptomatic"]):
        for kw in ["known", "history of", "detected", "symptomatic"]:
            if kw in claim_lower:
                idx = claim_lower.find(kw)
                start = max(0, idx - 20)
                end = min(len(claim), idx + 50)
                phrase = claim[start:end].strip()
                return f"{phrase}?"
    
    if "tolerating" in claim_lower:
        return "Is the patient tolerating the treatment?"
    if "indicated" in claim_lower:
        return "Is this treatment indicated for the patient?"
    
    return None


def generate_json_response(
    item: Dict,
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    source: str = "structured_db"
) -> Dict[str, Any]:
    """Generate structured JSON response with guideline references."""
    condition_name = item.get("condition_name", "Unknown Condition")
    criteria = item.get("criteria", "")
    source_page = item.get("source_page", "N/A")
    source_ref = item.get("source_reference_file", "")
    
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
        guideline_ref = arg.get("guideline_reference", source_ref)
        
        entry = {
            "claim": claim,
            "timing": timing if timing and timing != "As per guideline" else None,
            "category": category,
            "source_column": source_column,
            "guideline_reference": guideline_ref
        }
        
        if category == "ultrasound" or "scan" in claim.lower() or "doppler" in claim.lower():
            ultrasounds.append(entry)
        elif category == "test" or any(kw in claim.lower() for kw in ["blood", "fbc", "ogtt", "g&s", "tft", "urine", "bp", "ferritin", "hba1c", "antibody"]):
            tests.append(entry)
        elif any(kw in claim.lower() for kw in ["follow", "review", "next visit", "escalate", "refer"]):
            follow_up_items.append(entry)
        elif any(kw in claim.lower() for kw in ["if", "unless", "when", "consider", "assess"]):
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
                "original_claim": claim,
                "guideline_reference": guideline_ref
            })
        else:
            actions.append(entry)
        
        clarify_question = _claim_to_clarify_question(claim)
        if clarify_question:
            clarify_items.append({
                "question": clarify_question,
                "related_claim": claim,
                "timing": timing if timing and timing != "As per guideline" else None,
                "source_column": source_column,
                "guideline_reference": guideline_ref
            })
    
    if criteria:
        clarify_items.insert(0, {
            "question": f"Does the patient meet diagnostic criteria: {criteria}?",
            "related_claim": "criteria_verification",
            "timing": "At assessment",
            "source_column": "Criteria",
            "guideline_reference": source_ref
        })
    
    next_visits = [arg.get("timing", "") for arg in item.get("arguments", []) if arg.get("timing") and arg.get("timing") not in ["As per guideline", "N/A", ""]]
    upcoming = [t for t in next_visits if any(kw in t.lower() for kw in ["weeks", "booking", "next"])]
    next_review = upcoming[0] if upcoming else "Follow schedule per guidelines"
    
    rebuttals = [arg.get("rebuttal") for arg in item.get("arguments", []) if arg.get("rebuttal") and arg.get("rebuttal") != "None specified"]
    
    response = {
        "version": "7.1",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "query": query,
            "source": source,
            "condition_matched": condition_name,
            "patient_context": patient_context
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
                {"text": source_ref, "guideline_reference": source_ref} if source_ref and source_ref != "ANTENATAL CARE SCHEDULE.pdf" else None,
                {"text": f"ANTENATAL CARE SCHEDULE CHEAT SHEET (Page {source_page})", "guideline_reference": "ANTENATAL CARE SCHEDULE.pdf"}
            ],
            "actions": actions if actions else [],
            "tests": tests if tests else [],
            "ultrasound": ultrasounds if ultrasounds else [],
            "follow_up": follow_up_items if follow_up_items else [],
            "clarify": clarify_items if clarify_items else [],
            "decision_points": decision_points if decision_points else [],
            "plan": {
                "next_review": next_review,
                "exceptions": rebuttals if rebuttals else []
            }
        },
        "personalization": None
    }
    
    response["sections"]["guidelines"] = [g for g in response["sections"]["guidelines"] if g is not None]
    
    if patient_context:
        response["personalization"] = {
            "age_risk": patient_context.get("age", 0) >= 40,
            "bmi_category": categorize_bmi(patient_context.get("bmi")),
            "vte_risk": any(kw in " ".join(patient_context.get("medical_history", [])).lower() for kw in ["dvt", "thrombosis", "clot"]),
            "gestational_age": patient_context.get("gestational_age_weeks"),
            "triggered_conditions": get_triggered_conditions(patient_context)
        }
    
    return response


def query_vector_rag(
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    engine: GuidelineEngine = None
) -> Dict[str, Any]:
    """Fallback to vector RAG when no structured match found."""
    if engine is None:
        engine = GuidelineEngine()
    
    boosted_query = query
    boost_terms = []
    
    if any(term in query.lower() for term in ["preconception", "before pregnancy", "planning pregnancy"]):
        boost_terms.append("preconception folic acid")
    if any(term in query.lower() for term in ["aspirin", "prophylaxis", "pre-eclampsia"]):
        boost_terms.append("aspirin 150mg pre-eclampsia prophylaxis")
    if any(term in query.lower() for term in ["growth", "fgr", "sga", "small baby"]):
        boost_terms.append("fetal growth restriction scan monitoring")
    
    if boost_terms:
        boosted_query = " ".join(boost_terms) + " " + query
    
    docs = engine.vectordb.similarity_search(boosted_query, k=4)
    
    context_parts = []
    for d in docs:
        page = d.metadata.get('page', 'N/A')
        source = d.metadata.get('source', '')
        part = f"[Page {page}]" + (f" ({source})" if source else "") + f"\n{d.page_content}"
        context_parts.append(part)
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical assistant for antenatal care guidelines. Answer ONLY using the provided context. Answer in ENGLISH. If the answer is not in the context, say: 'I don't know based on the document.' Always include source page references when available."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])
    
    chain = prompt | engine.llm
    answer = chain.invoke({"question": query, "context": context})
    
    response = {
        "version": "7.1",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "query": query,
            "source": "vector_rag",
            "condition_matched": None,
            "patient_context": patient_context,
            "retrieved_pages": list(set([d.metadata.get('page', 'N/A') for d in docs]))
        },
        "answer": {
            "text": answer.content,
            "sources": [
                {
                    "page": d.metadata.get('page', 'N/A'),
                    "filename": Path(d.metadata.get('source', 'Unknown')).name
                }
                for d in docs
            ]
        },
        "condition": None,
        "sections": None,
        "personalization": None
    }
    
    return response


def create_error_response(error: str, query: str = None, patient_context: Dict = None) -> Dict:
    """Create standardized error response."""
    return {
        "version": "7.1",
        "timestamp": datetime.now().isoformat(),
        "error": {
            "type": "processing_error",
            "message": error,
            "query": query,
            "patient_context": patient_context
        },
        "condition": None,
        "sections": None,
        "personalization": None
    }


# =============================================================================
# OPTIONAL: CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What care is needed for anaemia?"
    
    patient_context = {
        "age": 42,
        "bmi": 45.0,
        "medical_history": ["previous history of DVT"]
    }
    

    result = query_guidelines(query=query, patient_context=patient_context)
    print(json.dumps(result, indent=2, ensure_ascii=False))