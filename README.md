# Antenatal RAG Assistant - AI Coding Guidelines

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system for antenatal care guidance. It combines structured argumentation databases with vector-based document retrieval to answer medical questions about pregnancy conditions.

## Architecture
- **Dual Database System**: 
  - Structured JSON database (`argument_db3.json`, ) for condition-specific recommendations, tests, ultrasound monitoring, and clinical rationale
  - Chroma vector database for full-text PDF retrieval using semantic search
- **Fallback Flow**: Questions first match structured conditions; unmatched queries use RAG with Ollama LLM
- **Context Expansion**: LinkRetrieverAgent automatically ingests referenced PDFs from retrieved chunks

## Key Components
- `ask.py`: Main query interface implementing the dual DB fallback
- `argument_db.json`: Structured antenatal condition data with recommendations/tests/ultrasound/reasons
- `create_vector_db.py`: Script to populate Chroma vector store from PDF chunks
- `populate_argument_db.py`: Heuristic parsing of extracted text into structured JSON format
- `hakathon_2026_chroma_store/`: Chroma vector store with PDF chunks
- `link_agent.py`: Agent for expanding context via linked documents
- `Initialize.py`: Centralized path configuration for all files and databases

## Integration Instructions
- `pipelone_agent.py`: Use this to integrate wth your system
- `example.py`: Example usage of the query interface with a sample patient case

## Development Workflow
1. **Setup**: Run `Initialize.py` to verify paths
2. **Data Preparation**:
   - Reset structured DB: `python reset_argument_db.py`
   - Setup structured DB Schema: `python setup_augment_db_schema.py`
   - Populate structured DB: `python populate_argument_db.py`
   - Create vector DB: `python create_vector_db.py`
   - Ingest PDF into structured DB: `python injest_pdf.py`
3. **Query**: `python ask.py` for interactive Q&A

## Code Patterns
- **Path Management**: Use `Initialize.py` for all file paths (PDF, DBs, outputs)
- **Embedding Model**: `HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")`
- **LLM**: `ChatOllama(model="gemma3:4b", temperature=0)`
- **PDF Loading**: `PDFPlumberLoader` for accurate text extraction
- **Text Splitting**: `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)`
- **Heuristic Parsing**: Classify text lines into categories using keyword matching (see `populate_argument_db_from_extracted_text.py`)

## Dependencies
- LangChain ecosystem (HuggingFace, Community, Core)
- ChromaDB for vector storage
- Ollama for local LLM inference
- NetworkX for argument graph construction
- PDFPlumber for PDF processing

## Data Flow
1. PDF → Text extraction → Structured parsing → JSON DB + Vector DB
2. Query → Match structured DB → Fallback to vector retrieval → LLM generation
3. Retrieved chunks → Link detection → PDF ingestion → Enhanced retrieval

## Testing
- Manual testing via `ask.py` interactive mode
- Validate structured DB queries with `query_argument_graph.py`
- Check vector retrieval with test queries in `create_vector_db.py`

## File Organization
- `Initialize.py`: Central path configuration
- `link_agent.py`: Context expansion logic