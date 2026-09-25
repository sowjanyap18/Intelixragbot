"""
query_test.py  (intelixsysRAG/chunking/query_test.py)

Quick sanity check: query the employee_kb collection you just built with
ingest.py, and see what chunks come back for a few sample questions.
This does NOT call Claude yet - it just proves retrieval is working.

Run:
    python query_test.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

SAMPLE_QUESTIONS = [
    "How many PTO days do I get and how far in advance do I need to request leave?",
    "Can I work fully remote?",
    "What happens if I lose my work laptop?",
    "Who do I contact if I experience harassment at work?",
    "How do I submit a travel expense for reimbursement?",
]


def main():
    print("Loading embedding model (same one used during ingestion)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="employee_kb", embedding_function=embed_fn)
    print(f"Loaded 'employee_kb' collection - {collection.count()} chunks total\n")

    for question in SAMPLE_QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}")
        results = collection.query(query_texts=[question], n_results=3)

        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            preview = doc.replace("\n", " ").strip()[:150]
            print(f"\n  [{meta['source']}]  (distance={dist:.3f})")
            print(f"  {preview}...")
        print()


if __name__ == "__main__":
    main()