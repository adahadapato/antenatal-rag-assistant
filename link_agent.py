# link_agent.py
import os
import re
import hashlib
import tempfile
from typing import List, Tuple, Optional

import requests
from langchain_community.document_loaders import PDFPlumberLoader


# ---------
# Utilities
# ---------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_links_from_docs(docs) -> Tuple[List[str], List[str]]:
    """
    Returns:
      urls: list of http(s) links found in retrieved chunks
      pdf_names: list of 'something.pdf' filenames found in retrieved chunks
    """
    joined = "\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])
    urls = re.findall(r"https?://\S+", joined)

    # Capture filenames like "FGR Guideline 2024 Hillingdon 2024.pdf"
    pdf_names = re.findall(r"\b[\w\-.() ]+\.pdf\b", joined, flags=re.IGNORECASE)

    # Clean up trailing punctuation from urls
    urls = [u.rstrip(").,;]}>\"'") for u in urls]

    # Deduplicate while preserving order
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedupe(urls), dedupe(pdf_names)


def _download_pdf_to_temp(url: str, timeout: int = 30) -> str:
    """
    Downloads a PDF URL to a temp file, returns path.
    Raises requests exceptions if it fails.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    # Basic check (some servers return HTML)
    content_type = (r.headers.get("Content-Type") or "").lower()
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        # still allow if it looks like a pdf, but warn via exception
        # (caller can decide what to do)
        pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def _load_pdf(path: str):
    loader = PDFPlumberLoader(path)
    return loader.load()


# -----------------------
# The "Second Agent" API
# -----------------------

class LinkRetrieverAgent:
    """
    Agent that expands retrieved context by following linked PDFs (URL or local filename),
    then ingests them into an existing Chroma vector store using vectordb.add_documents().
    """

    def __init__(
        self,
        linked_docs_dir: Optional[str] = None,
        state_dir: Optional[str] = None,
        max_new_docs_per_question: int = 2,
    ):
        """
        linked_docs_dir:
          If the document contains filenames (not URLs), the agent will look for them here.
          Example: r"C:\Projects\Hakathon\linked_docs"

        state_dir:
          Folder to store a tiny cache so the same URL/file isn't ingested repeatedly.
          Default: creates ".agent_state" in current working directory.
        """
        self.linked_docs_dir = linked_docs_dir
        self.max_new_docs_per_question = max_new_docs_per_question

        self.state_dir = state_dir or os.path.join(os.getcwd(), ".agent_state")
        os.makedirs(self.state_dir, exist_ok=True)
        self.cache_file = os.path.join(self.state_dir, "ingested_links.txt")
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                f.write("")

    def _is_ingested(self, key: str) -> bool:
        if not os.path.exists(self.cache_file):
            return False
        with open(self.cache_file, "r", encoding="utf-8") as f:
            return key.strip() in {line.strip() for line in f.readlines()}

    def _mark_ingested(self, key: str) -> None:
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(key.strip() + "\n")

    def expand_and_ingest(self, retrieved_docs, vectordb) -> List[str]:
        """
        Looks for links in retrieved_docs. If found, ingests up to max_new_docs_per_question
        PDFs into vectordb. Returns a list of actions taken (strings) for logging.
        """
        actions = []

        urls, pdf_names = extract_links_from_docs(retrieved_docs)

        candidates = []
        # prefer URLs first
        for u in urls:
            candidates.append(("url", u))
        for name in pdf_names:
            candidates.append(("file", name))

        added_count = 0

        for kind, value in candidates:
            if added_count >= self.max_new_docs_per_question:
                break

            if kind == "url":
                cache_key = "url:" + _sha1(value)
                if self._is_ingested(cache_key):
                    continue

                try:
                    tmp_pdf = _download_pdf_to_temp(value)
                    new_docs = _load_pdf(tmp_pdf)

                    # add metadata so you can see where it came from
                    for d in new_docs:
                        d.metadata["source_url"] = value
                        d.metadata["source_type"] = "url_pdf"

                    vectordb.add_documents(new_docs)
                    self._mark_ingested(cache_key)

                    actions.append(f"✅ Ingested linked PDF from URL: {value} ({len(new_docs)} pages)")
                    added_count += 1

                except Exception as e:
                    actions.append(f"⚠️ Failed to ingest URL '{value}': {e}")

            else:  # kind == "file"
                if not self.linked_docs_dir:
                    continue

                # Look up by exact filename in linked_docs_dir
                local_path = os.path.join(self.linked_docs_dir, value)
                if not os.path.exists(local_path):
                    # try a looser match (case-insensitive) if exact not found
                    matches = [f for f in os.listdir(self.linked_docs_dir) if f.lower() == value.lower()]
                    if matches:
                        local_path = os.path.join(self.linked_docs_dir, matches[0])

                if not os.path.exists(local_path):
                    continue

                cache_key = "file:" + _sha1(os.path.abspath(local_path))
                if self._is_ingested(cache_key):
                    continue

                try:
                    new_docs = _load_pdf(local_path)
                    for d in new_docs:
                        d.metadata["source_file"] = os.path.basename(local_path)
                        d.metadata["source_type"] = "local_pdf"

                    vectordb.add_documents(new_docs)
                    self._mark_ingested(cache_key)

                    actions.append(f"✅ Ingested linked local PDF: {local_path} ({len(new_docs)} pages)")
                    added_count += 1

                except Exception as e:
                    actions.append(f"⚠️ Failed to ingest local PDF '{local_path}': {e}")

        return actions