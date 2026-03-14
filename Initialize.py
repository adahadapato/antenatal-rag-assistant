
# Initialize.py - Path Configuration (Relative Paths)
# =============================================================================
# PURPOSE:
#   Centralized path configuration for the Antenatal Guidelines project.
#   Uses relative paths based on this file's location for portability.
# =============================================================================

from pathlib import Path
import os

# =============================================================================
# PROJECT ROOT DIRECTORY
# =============================================================================

# Resolve the absolute path of this file's parent directory
# This ensures paths work regardless of where the project is cloned/moved
BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# DATA & RESOURCE PATHS
# =============================================================================

# Input PDF document
doc_path = BASE_DIR / "ANTENATAL CARE SCHEDULE.pdf"

# Vector database (Chroma)
vector_db_path = BASE_DIR / "hakathon_2026_chroma_store"

# SQLite argumentation database
augmented_db_path = BASE_DIR / "augmented_db.db"

# JSON argumentation database (source of truth)
augmented_db_json = BASE_DIR / "augment_db4.json"

# Output files (logs, extracts, etc.)
output_file = BASE_DIR / "extracted_pdf_text.txt"

# Linked documents folder (for guideline PDFs)
linked_docs_dir = BASE_DIR / "linked_docs"

# Agent state cache
agent_state_dir = BASE_DIR / ".agent_state"

# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

def ensure_directories():
    """
    Create necessary directories if they don't exist.
    Call this once at application startup.
    """
    dirs_to_create = [
        vector_db_path,
        linked_docs_dir,
        agent_state_dir,
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

# Auto-create directories on import (optional)
# ensure_directories()

# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("✅ Initialize.py loaded successfully")
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📄 PDF Path: {doc_path}")
    print(f"🗄️  Vector DB: {vector_db_path}")
    print(f"🗄️  SQLite DB: {augmented_db_path}")
    print(f"📄 JSON DB: {augmented_db_json}")
    
    # Check if critical files exist
    if not doc_path.exists():
        print(f"⚠️  Warning: PDF not found at {doc_path}")
    if not augmented_db_json.exists():
        print(f"⚠️  Warning: JSON DB not found at {augmented_db_json}")