"""
ingest.py  (intelixsysRAG/chunking/ingest.py)

Reads .docx policy documents from ../data/employee (and, if present,
../data/client for public-facing content), chunks them, embeds them with a
local sentence-transformers model, and stores them in persistent Chroma
collections at ../chroma_db.

Folder layout this script expects:

  intelixsysRAG/
    chunking/
      ingest.py        <- this file
    data/
      employee/         *.docx  -> internal policy docs (employee_kb collection)
      client/            *.docx or *.txt -> public content (client_kb collection), optional
    chroma_db/          <- created automatically, persisted vector DB

Run from anywhere - paths are resolved relative to this file, not the cwd:
    python ingest.py
"""

import os
import glob
import docx
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../intelixsysRAG/chunking
PROJECT_ROOT = os.path.dirname(BASE_DIR)                 # .../intelixsysRAG
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def read_docx(path: str) -> str:
    """Extract text from a .docx file, including paragraphs and table cells,
    preserving blank lines between paragraphs so the chunker can split on them."""
    document = docx.Document(path)
    parts = []

    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
        else:
            parts.append("")  # preserve paragraph breaks for chunk_text()'s splitter

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    # collapse into paragraph-separated text (double newline = paragraph boundary)
    text = "\n".join(parts)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Sliding-window chunker that splits on paragraph boundaries where possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def load_docs(folder: str):
    """Load and chunk every .docx (and .txt, for convenience) file in a folder."""
    docs, metadatas, ids = [], [], []
    paths = sorted(glob.glob(os.path.join(folder, "*.docx")) + glob.glob(os.path.join(folder, "*.txt")))

    for path in paths:
        fname = os.path.basename(path)
        if fname.startswith("~$"):
            continue  # skip Word's temp lock files (e.g. ~$leave_policy.docx)

        if path.lower().endswith(".docx"):
            text = read_docx(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({"source": fname})
            ids.append(f"{fname}::chunk_{i}")

    return docs, metadatas, ids


def build_collection(client, collection_name: str, folder: str, embed_fn):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, embedding_function=embed_fn)

    if not os.path.isdir(folder):
        print(f"  (folder not found: {folder}, skipping)")
        return collection

    docs, metadatas, ids = load_docs(folder)
    if not docs:
        print(f"  (no .docx/.txt files found in {folder}, skipping)")
        return collection

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    n_files = len(set(m["source"] for m in metadatas))
    print(f"  Ingested {len(docs)} chunks from {n_files} file(s) into '{collection_name}'")
    return collection


def main():
    print("Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print("\nIngesting EMPLOYEE knowledge base (internal only)...")
    build_collection(client, "employee_kb", os.path.join(DATA_DIR, "employee"), embed_fn)

    print("\nIngesting CLIENT knowledge base (public, if present)...")
    build_collection(client, "client_kb", os.path.join(DATA_DIR, "client"), embed_fn)

    print("\nDone. Vector DB stored at:", CHROMA_PATH)


if __name__ == "__main__":
    main()