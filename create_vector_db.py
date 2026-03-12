import os
import Initialize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# 1. Use PyPDFLoader for PDF files
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

# 2. Updated import path for LangChain v1.x
#from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==========================================
# CONFIGURATION
# ==========================================
#folder_path = "./docs"  # Update this to your actual folder path
#pdf_filename = "ANTENATAL CARE SCHEDULE.pdf"
#doc_path = os.path.join(folder_path, pdf_filename)
#Initialize.ensure_paths()
doc_path = Initialize.doc_path
chroma_path = Initialize.vector_db_pathdb_path

# ==========================================
# 1) SETUP EMBEDDINGS
# ==========================================
# Note: "all-mpnet-base-v2" is the correct model name
print("⚙️ Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ==========================================
# 2) LOAD PDF DOCUMENT
# ==========================================
print(f"📄 Loading PDF: {doc_path}...")

if not os.path.exists(doc_path):
    raise FileNotFoundError(f"PDF not found at {doc_path}. Please check the path.")

#loader = PyPDFLoader(doc_path)
loader = PDFPlumberLoader(doc_path)
documents = loader.load()

print(f"✓ Loaded {len(documents)} pages from PDF")

# ==========================================
# 3) SPLIT DOCUMENTS
# ==========================================
# Chunk size 500 is good for dense medical schedules
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, 
                                               chunk_overlap=150,
                                                separators=["\n\n", "\n", "•", "-", "."])
splits = text_splitter.split_documents(documents)

print(f"✓ Split into {len(splits)} chunks")

# ==========================================
# 4) CREATE VECTOR STORE & SAVE
# ==========================================
print("🔗 Creating vector store & saving to disk...")
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=chroma_path
)

print(f"✓ Successfully stored {len(splits)} document chunks in '{chroma_path}'")

# ==========================================
# 5) TEST RETRIEVAL
# ==========================================
#retriever = vectordb.as_retriever(search_kwargs={"k": 2})
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,
        "k": 3
    }
)

# Try a query relevant to your document
query = "What is the schedule for antenatal visits?"
print(f"\n🔍 Query: {query}\n")

results = retriever.invoke(query)

print(f"📄 Retrieved Results:")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}) ---")
    print(f"{doc.page_content[:200]}...") # Print first 200 chars
    print("\n")