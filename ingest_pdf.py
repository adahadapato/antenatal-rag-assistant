# ingest_pdf.py - Optimized for Antenatal Care Guidelines
# =============================================================================
# PURPOSE:
#   Ingest PDF guidelines into Chroma vector store for RAG retrieval.
#   - Cleans OCR artifacts
#   - Preserves page metadata for source tracing
#   - Uses table-aware chunking for medical schedules
#   - Adds condition tags for filtered retrieval
#   - Handles empty metadata fields (ChromaDB compatibility)
#
# USAGE:
#   python ingest_pdf.py
#   Run before ask.py whenever PDF is updated
# =============================================================================

import os
import re
import Initialize
from pathlib import Path
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =============================================================================
# CONFIGURATION
# =============================================================================
doc_path = Initialize.doc_path
chroma_path = Initialize.vector_db_path

# Condition name mappings for auto-tagging (extracted from PDF Table of Contents)
CONDITION_KEYWORDS = {
    "advanced_maternal_age": ["advanced maternal age", "age > 40", "ama"],
    "anaemia": ["anaemia", "anemia", "hb <", "hemoglobin", "ferritin"],
    "bmi_high": ["bmi 35", "bmi > 40", "raised bmi", "obesity"],
    "bmi_low": ["low bmi", "bmi < 18.5"],
    "chicken_pox": ["chicken pox", "varicella", "vzv"],
    "chronic_hypertension": ["chronic hypertension", "htn", "high blood pressure"],
    "congenital_uterine_anomaly": ["congenital uterine anomaly", "uterine anomaly"],
    "epilepsy": ["epilepsy", "aed", "anti-epileptic", "seizure"],
    "reduced_fetal_movements": ["reduced fetal movements", "rfm", "decreased movements"],
    "fgr": ["fgr", "fetal growth restriction", "efw/ac < 3rd"],
    "fibroid_uterus": ["fibroid uterus", "fibroids > 3cm"],
    "gestational_diabetes": ["gestational diabetes", "gdm", "ogtt"],
    "herpes": ["herpes", "genital herpes", "aciclovir"],
    "hyperthyroid": ["hyperthyroid", "tsh < 0.1", "graves"],
    "hypothyroid": ["hypothyroid", "tsh > 10", "thyroxine"],
    "ibd": ["inflammatory bowel disease", "ibd", "crohn", "ulcerative colitis"],
    "ivf": ["ivf pregnancy", "ivf conception"],
    "lga": ["large baby", "lga", "efw/ac > 95th", "macrosomia"],
    "low_pappa": ["low pappa", "papp-a", "0.415 mom"],
    "maternal_antibodies": ["maternal antibodies", "anti-d", "anti-k", "alloantibodies"],
    "placenta_praevia": ["placenta praevia", "placenta previa", "low lying placenta"],
    "polyhydramnios": ["polyhydramnios", "afi > 25", "increased liquor"],
    "previous_cs": ["previous caesarean", "vbac", "prior cs"],
    "previous_congenital": ["previous congenital abnormality", "prior anomaly"],
    "previous_pet": ["previous pet", "previous pre-eclampsia", "prior hypertension"],
    "routine_care": ["routine care", "low risk", "uncomplicated"],
    "sga": ["sga baby", "efw/ac < 10th", "small for gestational age"],
    "sickle_cell": ["sickle cell disease", "scd", "haemoglobinopathy"],
    "smoker": ["smoker", "vaping", "smoking cessation"],
    "syphilis": ["syphilis", "treponemal", "rpr", "vdrl"],
    "thrombocytopenia": ["thrombocytopenia", "platelets < 150"],
    "twins_dcda": ["twins dcda", "dichorionic", "dcda"],
    "twins_mcda": ["twins mcda", "monochorionic", "mcda", "ttts"],
    "two_vessel_cord": ["two vessel cord", "single umbilical artery", "sua"],
    "preexisting_diabetes": ["type 1 diabetes", "type 2 diabetes", "pre-existing diabetes"],
    "aspirin_prophylaxis": ["aspirin prophylaxis", "aspirin 150mg"],
}


# =============================================================================
# HELPER: CLEAN OCR ARTIFACTS
# =============================================================================
def clean_ocr_text(text: str) -> str:
    """
    Clean common OCR artifacts from PDF text.
    FIX: Apply dictionary replacements FIRST, then targeted regex.
    """
    # STEP 1: Apply dictionary replacements FIRST (exact matches)
    replacements = {
        'p re g nanc y': 'pregnancy',
        'w ee k s': 'weeks',
        'b lood': 'blood',
        's can': 'scan',
        't es t': 'test',
        'g ui de lin e': 'guideline',
        'H yp e rt e n s ion': 'Hypertension',
        'Di abe tes': 'Diabetes',
        'Pl ace nt a': 'Placenta',
        'Pr ae vi a': 'Praevia',
        'F e t a l': 'Fetal',
        'G rowth': 'Growth',
        'S FH': 'SFH',
        'I O L': 'IOL',
        'C S': 'CS',
        'L SCS': 'LSCS',
        'F G R': 'FGR',
        'SG A': 'SGA',
        'L G A': 'LGA',
        'T TTS': 'TTTS',
        'M C DA': 'MCDA',
        'D C DA': 'DCDA',
        'U A': 'UA',
        'PI': 'PI',
        'EFW': 'EFW',
        'AC': 'AC',
        'A FI': 'AFI',
        'DVP': 'DVP',
        'TFT': 'TFT',
        'FBC': 'FBC',
        'G & S': 'G&S',
        'BP': 'BP',
        'CTG': 'CTG',
        'USS': 'USS',
        'FMU': 'FMU',
        'ANC': 'ANC',
        'MDT': 'MDT',
        'VTE': 'VTE',
        'PET': 'PET',
        'GDM': 'GDM',
        'OGTT': 'OGTT',
        'HbA1c': 'HbA1c',
        'LMWH': 'LMWH',
        'R COG': 'RCOG',
        'M C A': 'MCA',
        'P S V': 'PSV',
        'T R Ab': 'TRAb',
        'I G': 'Ig',
        'E I A': 'EIA',
        'C LIA': 'CLIA',
        'T PPA': 'TPPA',
        'V DRL': 'VDRL',
        'R P R': 'RPR',
        'D A T': 'DAT',
        'A P S': 'APS',
        'A N A': 'ANA',
        'L F T': 'LFT',
        'U & E': 'U&E',
        'M S U': 'MSU',
        'B M S': 'BMs',
        'D S M': 'DSM',
        'M O B': 'MOB',
        'O A S I': 'OASI',
        'P T B': 'PTB',
        'E C H O': 'ECHO',
        'S U D E P': 'SUDEP',
        'V Z V': 'VZV',
        'H I V': 'HIV',
        'H e p B': 'Hep B',
        'H e p C': 'Hep C',
        'B A S HH': 'BASHH',
        'G U M': 'GUM',
        'T O R CH': 'TORCH',
        'C S T': 'CST',
        'H R A N': 'HRAN',
        'M C D A': 'MCDA',
        'D C D A': 'DCDA',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # STEP 2: Targeted regex for remaining single-letter spacing artifacts
    # Only fix patterns like "a b c" where each is a single letter with spaces
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)
    
    # STEP 3: Normalize multiple spaces to single space (preserve sentence structure)
    text = re.sub(r'\s{2,}', ' ', text)
    
    # STEP 4: Fix common PDF extraction issues
    text = text.replace(' - ', ' – ')  # Fix dash spacing
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)  # Fix number ranges
    
    return text.strip()


# =============================================================================
# HELPER: AUTO-TAG CONDITIONS
# =============================================================================
def extract_condition_tags(text: str) -> List[str]:
    """
    Extract condition tags from text based on keyword matching.
    Enables filtered retrieval in ask.py.
    """
    tags = []
    text_lower = text.lower()
    
    for condition, keywords in CONDITION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(condition)
    
    return tags


# =============================================================================
# HELPER: EXTRACT GUIDELINE REFERENCES
# =============================================================================
def extract_guideline_refs(text: str) -> List[str]:
    """
    Extract guideline reference codes from text (e.g., NG133, V5.1, etc.)
    """
    refs = []
    
    # Pattern for PDF filenames (e.g., 202511211517060.Anaemia...pdf)
    pattern = r'\d{16,}\.[\w\s]+\.pdf'
    matches = re.findall(pattern, text, re.IGNORECASE)
    refs.extend(matches)
    
    # Pattern for versioned PDFs (e.g., V5.1.pdf)
    version_pattern = r'V\d+\.\d+'
    refs.extend(re.findall(version_pattern, text))
    
    # Pattern for Hillingdon guidelines
    hillingdon_pattern = r'Hillingdon\s+\d{4}'
    refs.extend(re.findall(hillingdon_pattern, text, re.IGNORECASE))
    
    # Pattern for NG guidelines (e.g., NG133 1.1.3)
    ng_pattern = r'NG\d+\s+\d+\.\d+\.\d+'
    refs.extend(re.findall(ng_pattern, text))
    
    return refs


# =============================================================================
# 1) SETUP EMBEDDINGS
# =============================================================================
print("⚙️ Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


# =============================================================================
# 2) LOAD PDF DOCUMENT
# =============================================================================
print(f"📄 Loading PDF: {doc_path}...")

if not os.path.exists(doc_path):
    raise FileNotFoundError(f"PDF not found at {doc_path}. Please check the path.")

try:
    loader = PDFPlumberLoader(doc_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages from PDF")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    # Fallback: try PyPDFLoader if PDFPlumber fails
    from langchain_community.document_loaders import PyPDFLoader
    print("🔄 Trying PyPDFLoader as fallback...")
    loader = PyPDFLoader(doc_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages with PyPDFLoader")


# =============================================================================
# 3) ADD METADATA (PAGE + SOURCE + TAGS)
# =============================================================================
source_filename = Path(doc_path).name

print("🏷️  Adding metadata and cleaning text...")
for i, doc in enumerate(documents):
    doc.metadata['page'] = i + 1  # 1-indexed for human readability
    doc.metadata['source'] = source_filename
    doc.metadata['filename'] = source_filename
    doc.metadata['doc_type'] = 'antenatal_guideline'
    doc.metadata['institution'] = 'Hillingdon'  # Based on PDF content
    
    # Clean OCR artifacts
    doc.page_content = clean_ocr_text(doc.page_content)
    
    # Extract condition tags for filtered retrieval
    condition_tags = extract_condition_tags(doc.page_content)
    if condition_tags:  # FIX: Only add if non-empty (ChromaDB requirement)
        doc.metadata['condition_tags'] = condition_tags
    
    # Extract guideline references
    guideline_refs = extract_guideline_refs(doc.page_content)
    if guideline_refs:  # FIX: Only add if non-empty (ChromaDB requirement)
        doc.metadata['guideline_refs'] = guideline_refs


# =============================================================================
# 4) SPLIT DOCUMENTS (TABLE-AWARE)
# =============================================================================
print("✂️  Splitting documents...")

# For table-heavy PDFs: larger chunks + table-friendly separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,          # Larger to preserve table row context
    chunk_overlap=250,        # More overlap for continuity across splits
    separators=[
        "\n\n",               # Paragraph breaks
        "\n",                 # Line breaks
        "•",                  # Bullet points
        "|",                  # Table column separators (preserve table structure)
        ".",                 # Sentence ends (last resort)
    ]
)

splits = text_splitter.split_documents(documents)

# Ensure all splits inherit metadata (and remove empty lists)
for split in splits:
    split.metadata.setdefault('page', 'N/A')
    split.metadata.setdefault('source', source_filename)
    split.metadata.setdefault('filename', source_filename)
    split.metadata.setdefault('doc_type', 'antenatal_guideline')
    split.metadata.setdefault('institution', 'Hillingdon')
    
    # FIX: Remove empty list metadata fields (ChromaDB doesn't allow empty lists)
    if 'condition_tags' in split.metadata and not split.metadata['condition_tags']:
        del split.metadata['condition_tags']
    if 'guideline_refs' in split.metadata and not split.metadata['guideline_refs']:
        del split.metadata['guideline_refs']

print(f"✓ Split into {len(splits)} chunks")


# =============================================================================
# 5) CREATE VECTOR STORE & SAVE
# =============================================================================
print("🔗 Creating vector store & saving to disk...")

# Delete existing Chroma DB to avoid stale data
if os.path.exists(chroma_path):
    import shutil
    shutil.rmtree(chroma_path)
    print(f"🗑️  Cleared existing vector store at {chroma_path}")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=chroma_path
)

print(f"✓ Successfully stored {len(splits)} document chunks in '{chroma_path}'")


# =============================================================================
# 6) TEST RETRIEVAL
# =============================================================================
print("\n" + "=" * 70)
print("🧪 TESTING RETRIEVAL")
print("=" * 70)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Test queries based on your PDF content
test_queries = [
    "What is the anaemia follow-up schedule?",
    "When should I scan for gestational diabetes?",
    "What is the aspirin prophylaxis guideline?",
    "What care is needed for twins MCDA?",
    "What is the schedule for reduced fetal movements?",
    "When to refer for placenta praevia?",
    "What is the management for previous pre-eclampsia?",
]

for query in test_queries:
    print(f"\n🔍 Query: {query}\n")
    results = retriever.invoke(query)
    
    if not results:
        print("⚠️  No results retrieved")
        continue
    
    for i, doc in enumerate(results):
        page = doc.metadata.get('page', 'N/A')
        source = doc.metadata.get('source', 'Unknown')
        tags = doc.metadata.get('condition_tags', [])
        refs = doc.metadata.get('guideline_refs', [])
        
        print(f"--- Result {i+1} (Page {page} • {source}) ---")
        if tags:
            print(f"    Tags: {', '.join(tags)}")
        if refs:
            print(f"    References: {', '.join(refs)}")
        # Print first 400 chars for better context visibility
        content_preview = doc.page_content[:400].replace('\n', ' ')
        print(f"    {content_preview}...")
        print()
    
    print("-" * 70)


# =============================================================================
# 7) SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("📊 INGESTION SUMMARY")
print("=" * 70)
print(f"Source PDF: {source_filename}")
print(f"Total pages loaded: {len(documents)}")
print(f"Total chunks created: {len(splits)}")
if len(documents) > 0:
    print(f"Average chunks per page: {len(splits) / len(documents):.1f}")
print(f"Vector store location: {chroma_path}")
print(f"Embedding model: all-mpnet-base-v2")
print(f"Chunk size: 1500, Overlap: 250")

# Count condition tags
all_tags = []
for split in splits:
    tags = split.metadata.get('condition_tags', [])
    if tags:
        all_tags.extend(tags)
tag_counts = {}
for tag in all_tags:
    tag_counts[tag] = tag_counts.get(tag, 0) + 1

print(f"\n🏷️  Top condition tags found:")
for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {tag}: {count} chunks")

print("=" * 70)
print("✅ Ingestion complete! Run ask.py to query the guidelines.")
print("=" * 70)