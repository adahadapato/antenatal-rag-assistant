# =============================================================================
# ANTEATAL GUIDELINES QUERY MODULE
# =============================================================================
# VERSION: 7.0
# AUTHOR: Clinical Decision Support Team
# CREATED: 2025-01-21
# LAST UPDATED: 2025-01-21
#
# PURPOSE:
#   This module provides programmatic access to antenatal care guidelines.
#   It queries a structured database of 37 pregnancy conditions and returns
#   evidence-based recommendations for clinical decision support.
#
# INTEGRATION:
#   Import as a Python module and call functions directly. No CLI, no files.
#   Designed for integration into larger healthcare software systems.
#
# DATA SOURCE:
#   - argument_db.json: 37 antenatal conditions extracted from PDF guidelines
#   - Chroma DB: Vector store for fallback semantic search
#   - Ollama LLM: Local language model for unstructured queries
#
# USAGE EXAMPLE:
#   from pipeline_agent import query_guidelines
#   
#   result = query_guidelines(
#       query="What care is needed for anaemia?",
#       patient_context={"age": 42, "bmi": 45.0, "medical_history": ["DVT"]}
#   )
#   
#   print(result["condition"]["name"])
#   print(result["sections"]["actions"])
#
# SUPPORT:
#   For issues or questions, contact the development team.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Third-party imports (LangChain ecosystem)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Optional: fuzzy matching for flexible condition name matching
try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    # Fuzzy matching disabled - will use exact matching only


# =============================================================================
# GLOBAL STATE: GUIDELINE ENGINE (SINGLETON)
# =============================================================================

class GuidelineEngine:
    """
    Singleton engine for antenatal guideline queries.
    
    This class manages all database connections and loaded data. It uses the
    Singleton pattern to ensure databases are loaded only once, then reused
    across multiple queries for optimal performance.
    
    Attributes:
        argument_data (List[Dict]): All 37 conditions from argument_db.json
        condition_lookup (Dict): Fast lookup by condition name (lowercase)
        condition_names (List[str]): List of all condition names
        _vectordb (Chroma): Vector database for semantic search (lazy loaded)
        _embeddings (HuggingFaceEmbeddings): Embedding model (lazy loaded)
        _llm (ChatOllama): Language model for RAG fallback (lazy loaded)
    
    Example:
        >>> engine = GuidelineEngine()  # Loads on first call
        >>> engine2 = GuidelineEngine()  # Returns same instance
        >>> engine is engine2
        True
    """
    
    # Class-level singleton state
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """
        Create or return the singleton instance.
        
        This ensures only one GuidelineEngine exists in memory, preventing
        duplicate database loads.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the engine (only runs once due to singleton pattern).
        
        Loads:
        1. Argumentation database (JSON) - loaded immediately
        2. Vector database (Chroma) - lazy loaded on first query
        3. LLM (Ollama) - lazy loaded on first RAG query
        """
        if GuidelineEngine._initialized:
            return  # Already initialized, skip
        
        # Load argumentation database immediately
        self.argument_data = self._load_argument_db()
        
        # Build fast lookup dictionaries
        self.condition_lookup = {
            item["condition_name"].lower(): item 
            for item in self.argument_data
        }
        self.condition_names = list(self.condition_lookup.keys())
        
        # Lazy-loaded components (initialized on first use)
        self._vectordb = None
        self._embeddings = None
        self._llm = None
        
        # Mark as initialized
        GuidelineEngine._initialized = True
    
    def _load_argument_db(self) -> List[Dict]:
        """
        Load argumentation database from Initialize.py configuration.
        
        Returns:
            List[Dict]: List of 37 condition records with arguments
            
        Raises:
            FileNotFoundError: If argument_db.json not found
            json.JSONDecodeError: If JSON is malformed
        """
        import Initialize
        
        with open(Initialize.augmented_db_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle wrapper object (some JSON versions have {"arguments": [...]})
        if isinstance(data, dict) and "arguments" in data:
            return data["arguments"]
        return data
    
    @property
    def vectordb(self):
        """
        Lazy load vector database on first access.
        
        Returns:
            Chroma: Vector database for semantic search
            
        Note:
            Vector DB is only loaded when needed (RAG fallback queries).
            This reduces startup time for structured-only queries.
        """
        if self._vectordb is None:
            import Initialize
            self._embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            self._vectordb = Chroma(
                persist_directory=Initialize.vector_db_path,
                embedding_function=self._embeddings
            )
        return self._vectordb
    
    @property
    def llm(self):
        """
        Lazy load Ollama LLM on first access.
        
        Returns:
            ChatOllama: Local language model for RAG fallback
            
        Note:
            LLM is only loaded when structured DB has no match.
            Requires Ollama service running locally with gemma3:4b model.
        """
        if self._llm is None:
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
    
    This is the primary entry point for all guideline queries. It accepts a
    natural language question and optional patient context, then returns
    structured recommendations.
    
    Args:
        query (str): User's question about antenatal care.
            Examples:
            - "What care is needed for anaemia?"
            - "When should I scan for gestational diabetes?"
            - "What is the plan for previous pre-eclampsia?"
        
        patient_context (Dict, optional): Patient data for personalization.
            Schema:
            {
                "age": int,                    # Patient age in years
                "gestational_age_weeks": int,  # Current pregnancy week (or null)
                "bmi": float,                  # Body Mass Index
                "parity": int,                 # Number of previous births
                "medical_history": List[str],  # Past medical conditions
                "obstetric_history": List[str],# Past pregnancy complications
                "current_medications": List[str] # Current medications
            }
            Example:
            {
                "age": 42,
                "gestational_age_weeks": null,
                "bmi": 45.0,
                "parity": 2,
                "medical_history": ["previous history of DVT"],
                "obstetric_history": [],
                "current_medications": []
            }
    
    Returns:
        Dict[str, Any]: Structured guideline response with the following schema:
        {
            "version": str,              # API version (e.g., "7.0")
            "timestamp": str,            # ISO 8601 timestamp
            "metadata": {
                "query": str,            # Original query string
                "source": str,           # "structured_db" or "vector_rag"
                "condition_matched": str, # Matched condition name
                "patient_context": Dict  # Input patient context (echoed)
            },
            "condition": {
                "id": str,               # Condition ID (e.g., "ANA-01")
                "name": str,             # Condition name
                "criteria": str,         # Diagnostic criteria
                "source": {
                    "page": int,         # PDF page number
                    "section": str,      # PDF section name
                    "reference_file": str # Linked guideline file
                }
            },
            "sections": {
                "guidelines": List[Dict], # Source guideline references
                "actions": List[Dict],    # Recommended interventions
                "tests": List[Dict],      # Lab/diagnostic tests
                "ultrasound": List[Dict], # Scan schedule
                "follow_up": List[Dict],  # Follow-up appointments
                "decision_points": List[Dict], # IF-THEN clinical logic
                "plan": {
                    "next_review": str,   # Next appointment timing
                    "exceptions": List[str] # Clinical exceptions/notes
                }
            },
            "personalization": {         # Only if patient_context provided
                "age_risk": bool,        # True if age >= 40
                "bmi_category": str,     # BMI risk category
                "vte_risk": bool,        # True if DVT/thrombosis history
                "gestational_age": int,  # Current pregnancy week
                "triggered_conditions": List[str] # Auto-triggered conditions
            }
        }
    
    Raises:
        None: All errors are caught and returned as error response dict
    
    Example:
        >>> result = query_guidelines(
        ...     query="What care is needed for anaemia?",
        ...     patient_context={"age": 42, "bmi": 45.0, "medical_history": ["DVT"]}
        ... )
        >>> print(result["condition"]["name"])
        Multiple Conditions: Anaemia, Advanced maternal age-> 40, BMI > 40
        >>> print(result["sections"]["actions"])
        [{'claim': 'If anaemia start oral iron...', 'timing': 'Upon diagnosis', ...}]
        >>> print(result["personalization"]["age_risk"])
        True
    """
    engine = GuidelineEngine()
    
    try:
        # Step 1: Get triggered conditions from patient context
        # (e.g., age >= 40 → Advanced Maternal Age condition)
        triggered = get_triggered_conditions(patient_context) if patient_context else []
        
        # Step 2: Find matching condition(s) in structured database
        matched_item = find_condition_in_question(
            query, 
            triggered_conditions=triggered,
            condition_lookup=engine.condition_lookup,
            condition_names=engine.condition_names
        )
        
        if matched_item:
            # Step 3a: Return structured database response
            response = generate_json_response(
                item=matched_item,
                query=query,
                patient_context=patient_context,
                source="structured_db"
            )
        else:
            # Step 3b: Fallback to vector RAG (semantic search + LLM)
            response = query_vector_rag(
                query=query,
                patient_context=patient_context,
                engine=engine
            )
        
        return response
        
    except Exception as e:
        # Return standardized error response (never raise exceptions)
        return create_error_response(error=str(e), query=query, patient_context=patient_context)


# =============================================================================
# HELPER: GET TRIGGERED CONDITIONS FROM PATIENT CONTEXT
# =============================================================================

def get_triggered_conditions(patient: Optional[Dict[str, Any]]) -> List[str]:
    """
    Determine which conditions are triggered by patient parameters.
    
    This function analyzes patient demographics and medical history to
    auto-include relevant conditions. For example, a patient with BMI > 40
    automatically triggers the "BMI > 40" condition, even if not explicitly
    mentioned in the query.
    
    Args:
        patient (Dict): Patient context dictionary from query_guidelines()
    
    Returns:
        List[str]: List of condition_name values that should be auto-included
        
    Trigger Rules:
        - Age >= 40 → "advanced maternal age-> 40"
        - BMI 35-39.9 → "bmi 35-39.9"
        - BMI >= 40 → "bmi > 40"
        - BMI < 18.5 → "low bmi < 18.5"
        - Medical history contains "dvt"/"thrombosis"/"clot" → "bmi > 40" (VTE risk)
        - Medical history contains "diabetes"/"type 1"/"type 2" → "type 1 and 2 diabetes"
        - Medical history contains "hypertension"/"htn" → "chronic hypertension"
        - Medical history contains "epilepsy"/"seizure" → "epilepsy"
        - Medical history contains "sickle"/"haemoglobinopathy" → "sickle cell disease"
        - Obstetric history contains "pre-eclampsia"/"pet" → "previous pet"
        - Obstetric history contains "caesarean"/"c-section" → "previous caesarean section"
        - Obstetric history contains "twins"/"multiple" → "twins"
    
    Example:
        >>> patient = {"age": 42, "bmi": 45.0, "medical_history": ["DVT"]}
        >>> get_triggered_conditions(patient)
        ['advanced maternal age-> 40', 'bmi > 40', 'bmi > 40']
    """
    if not patient:
        return []
    
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
    
    # Medical history triggers (case-insensitive keyword matching)
    medical_history = " ".join(patient.get("medical_history", [])).lower()
    if "dvt" in medical_history or "thrombosis" in medical_history or "clot" in medical_history:
        triggered.append("bmi > 40")  # VTE risk assessment
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


# =============================================================================
# HELPER: FIND MATCHING CONDITION IN DATABASE
# =============================================================================

def find_condition_in_question(
    question: str,
    triggered_conditions: List[str] = None,
    condition_lookup: Dict = None,
    condition_names: List[str] = None
) -> Optional[Dict]:
    """
    Find matching condition(s) from question and triggered conditions.
    
    Uses a three-tier matching approach:
    1. Exact substring match (fastest)
    2. Triggered conditions from patient context
    3. Fuzzy matching (if fuzzywuzzy installed)
    4. Synonym mapping (medical term variations)
    
    Args:
        question (str): User's query string
        triggered_conditions (List[str], optional): Conditions from patient context
        condition_lookup (Dict, optional): Pre-built condition lookup (for testing)
        condition_names (List[str], optional): Pre-built condition names (for testing)
    
    Returns:
        Optional[Dict]: Matching condition record, merged condition (if multiple),
                       or None if no match found
    
    Example:
        >>> find_condition_in_question("What care for anaemia?")
        {'condition_id': 'ANA-01', 'condition_name': 'Anaemia', ...}
        
        >>> find_condition_in_question("high blood pressure in pregnancy")
        {'condition_id': 'HTN-01', 'condition_name': 'Chronic Hypertension', ...}
    """
    # Lazy load engine if lookup tables not provided
    if condition_lookup is None or condition_names is None:
        engine = GuidelineEngine()
        condition_lookup = engine.condition_lookup
        condition_names = engine.condition_names
    
    q_lower = question.lower()
    matched_items = []
    
    # Tier 1: Exact substring match (fastest path)
    for condition_name in condition_names:
        if condition_name in q_lower:
            matched_items.append(condition_lookup[condition_name])
    
    # Tier 2: Add triggered conditions from patient context
    if triggered_conditions:
        for trigger in triggered_conditions:
            for condition_name in condition_names:
                if trigger.lower() in condition_name and condition_lookup[condition_name] not in matched_items:
                    matched_items.append(condition_lookup[condition_name])
    
    # Tier 3: Fuzzy matching (if available and no exact matches)
    if FUZZY_AVAILABLE and not matched_items:
        match = process.extractOne(q_lower, condition_names, score_cutoff=60)
        if match:
            matched_items.append(condition_lookup[match[0]])
    
    # Tier 4: Synonym mapping (medical term variations)
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
    
    # Return single match, merged multiple matches, or None
    if len(matched_items) == 1:
        return matched_items[0]
    elif len(matched_items) > 1:
        return merge_conditions(matched_items)
    
    return None


# =============================================================================
# HELPER: MERGE MULTIPLE CONDITIONS
# =============================================================================

def merge_conditions(condition_list: List[Dict]) -> Dict:
    """
    Merge multiple condition records into a single response.
    
    Used when a patient triggers multiple conditions (e.g., Anaemia + BMI > 40 +
    Advanced Maternal Age). This combines all recommendations while deduplicating
    identical claims.
    
    Args:
        condition_list (List[Dict]): List of condition records to merge
    
    Returns:
        Dict: Merged condition record with combined arguments
        
    Example:
        >>> conditions = [anaemia_record, bmi_record]
        >>> merge_conditions(conditions)
        {
            "condition_id": "MERGED-ANA-01-BMI-02",
            "condition_name": "Multiple Conditions: Anaemia, BMI > 40",
            "arguments": [...]  # Combined from both conditions
        }
    """
    merged = {
        "condition_id": "MERGED-" + "-".join([c["condition_id"] for c in condition_list]),
        "condition_name": "Multiple Conditions: " + ", ".join([c["condition_name"] for c in condition_list]),
        "criteria": "Patient meets criteria for: " + "; ".join([c.get("criteria", "N/A") for c in condition_list]),
        "source_page": ", ".join([str(c.get("source_page", "N/A")) for c in condition_list]),
        "source_reference_file": ", ".join([c.get("source_reference_file", "N/A") for c in condition_list]),
        "arguments": []
    }
    
    # Combine all arguments, deduplicating by (claim, timing) pair
    seen_claims = set()
    for cond in condition_list:
        for arg in cond.get("arguments", []):
            claim_key = (arg.get("claim"), arg.get("timing"))
            if claim_key not in seen_claims:
                seen_claims.add(claim_key)
                merged["arguments"].append(arg)
    
    return merged


# =============================================================================
# HELPER: GENERATE STRUCTURED JSON RESPONSE
# =============================================================================

def generate_json_response(
    item: Dict,
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    source: str = "structured_db"
) -> Dict[str, Any]:
    """
    Generate structured JSON response for integration.
    
    Converts a condition record into the standardized API response format
    with categorized sections (actions, tests, ultrasound, etc.).
    
    Args:
        item (Dict): Condition record from argument database
        query (str): Original user query
        patient_context (Dict, optional): Patient data for personalization
        source (str): Response source ("structured_db" or "vector_rag")
    
    Returns:
        Dict[str, Any]: Standardized response schema (see query_guidelines docstring)
    """
    condition_name = item.get("condition_name", "Unknown Condition")
    criteria = item.get("criteria", "")
    source_page = item.get("source_page", "N/A")
    source_ref = item.get("source_reference_file", "")
    
    # Categorize arguments by type
    actions = []
    tests = []
    ultrasounds = []
    follow_up_items = []
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
        
        # Categorize by type (category field + keyword matching)
        if category == "ultrasound" or "scan" in claim.lower() or "doppler" in claim.lower():
            ultrasounds.append(entry)
        elif category == "test" or any(kw in claim.lower() for kw in ["blood", "fbc", "ogtt", "g&s", "tft", "urine", "bp", "ferritin", "hba1c", "antibody"]):
            tests.append(entry)
        elif any(kw in claim.lower() for kw in ["follow", "review", "next visit", "escalate", "refer"]):
            follow_up_items.append(entry)
        elif any(kw in claim.lower() for kw in ["if", "unless", "when", "consider", "assess"]):
            # Extract IF-THEN decision points
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
    
    # Determine next review timing
    next_visits = [arg.get("timing", "") for arg in item.get("arguments", []) if arg.get("timing") and arg.get("timing") not in ["As per guideline", "N/A", ""]]
    upcoming = [t for t in next_visits if any(kw in t.lower() for kw in ["weeks", "booking", "next"])]
    next_review = upcoming[0] if upcoming else "Follow schedule per guidelines"
    
    # Collect exceptions/rebuttals
    rebuttals = [arg.get("rebuttal") for arg in item.get("arguments", []) if arg.get("rebuttal") and arg.get("rebuttal") != "None specified"]
    
    # Build response structure
    response = {
        "version": "7.0",
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
                {"text": source_ref} if source_ref and source_ref != "ANTENATAL CARE SCHEDULE.pdf" else None,
                {"text": f"ANTENATAL CARE SCHEDULE CHEAT SHEET (Page {source_page})"}
            ],
            "actions": actions if actions else [],
            "tests": tests if tests else [],
            "ultrasound": ultrasounds if ultrasounds else [],
            "follow_up": follow_up_items if follow_up_items else [],
            "decision_points": decision_points if decision_points else [],
            "plan": {
                "next_review": next_review,
                "exceptions": rebuttals if rebuttals else []
            }
        },
        "personalization": None
    }
    
    # Clean up None values in guidelines
    response["sections"]["guidelines"] = [g for g in response["sections"]["guidelines"] if g is not None]
    
    # Add personalization flags if patient context provided
    if patient_context:
        response["personalization"] = {
            "age_risk": patient_context.get("age", 0) >= 40,
            "bmi_category": categorize_bmi(patient_context.get("bmi")),
            "vte_risk": any(kw in " ".join(patient_context.get("medical_history", [])).lower() for kw in ["dvt", "thrombosis", "clot"]),
            "gestational_age": patient_context.get("gestational_age_weeks"),
            "triggered_conditions": get_triggered_conditions(patient_context)
        }
    
    return response


# =============================================================================
# HELPER: QUERY VECTOR RAG (FALLBACK)
# =============================================================================

def query_vector_rag(
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    engine: GuidelineEngine = None
) -> Dict[str, Any]:
    """
    Fallback to vector RAG when no structured match found.
    
    Uses semantic search (embeddings) to find relevant PDF chunks, then
    generates an answer using the local Ollama LLM.
    
    Args:
        query (str): User's question
        patient_context (Dict, optional): Patient data for context
        engine (GuidelineEngine, optional): Pre-loaded engine instance
    
    Returns:
        Dict[str, Any]: RAG response with answer text and sources
    """
    if engine is None:
        engine = GuidelineEngine()
    
    # Query boosting for better retrieval
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
    
    # Retrieve from vector DB
    docs = engine.vectordb.similarity_search(boosted_query, k=4)
    
    # Build context string
    context_parts = []
    for d in docs:
        page = d.metadata.get('page', 'N/A')
        source = d.metadata.get('source', '')
        part = f"[Page {page}]" + (f" ({source})" if source else "") + f"\n{d.page_content}"
        context_parts.append(part)
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer with LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical assistant for antenatal care guidelines. Answer ONLY using the provided context. Answer in ENGLISH. If the answer is not in the context, say: 'I don't know based on the document.' Always include source page references when available."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])
    
    chain = prompt | engine.llm
    answer = chain.invoke({"question": query, "context": context})
    
    # Build response
    response = {
        "version": "7.0",
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


# =============================================================================
# HELPER: CATEGORIZE BMI
# =============================================================================

def categorize_bmi(bmi: float) -> str:
    """
    Categorize BMI for risk stratification.
    
    Args:
        bmi (float): Body Mass Index value
    
    Returns:
        str: BMI category string
        
    Categories:
        - underweight: BMI < 18.5
        - normal: BMI 18.5-24.9
        - overweight: BMI 25-29.9
        - obese_class_1_2: BMI 30-39.9
        - obese_class_3: BMI >= 40
    """
    if bmi is None:
        return "unknown"
    elif bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    elif bmi < 40:
        return "obese_class_1_2"
    else:
        return "obese_class_3"


# =============================================================================
# HELPER: CREATE ERROR RESPONSE
# =============================================================================

def create_error_response(error: str, query: str = None, patient_context: Dict = None) -> Dict:
    """
    Create standardized error response.
    
    Ensures all errors return valid JSON (never raise exceptions).
    
    Args:
        error (str): Error message
        query (str, optional): Original query that caused error
        patient_context (Dict, optional): Patient context from request
    
    Returns:
        Dict[str, Any]: Error response with consistent schema
    """
    return {
        "version": "7.0",
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
# OPTIONAL: CLI FOR TESTING (NOT USED IN INTEGRATION)
# =============================================================================

if __name__ == "__main__":
    """
    Test harness for direct module execution.
    
    This block only runs when executing the file directly:
        python antenatal_guidelines.py
    
    Not used when importing as a module.
    """
    import sys
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What care is needed for anaemia?"
    
    # Test patient context
    patient_context = {
        "age": 42,
        "bmi": 45.0,
        "medical_history": ["previous history of DVT"]
    }
    
    # Execute query and print result
    result = query_guidelines(query=query, patient_context=patient_context)
    print(json.dumps(result, indent=2, ensure_ascii=False))