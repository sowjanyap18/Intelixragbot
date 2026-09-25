"""
chatbot.py  (intelixsysRAG/chunking/chatbot.py)

The generation layer. Takes a question, retrieves the top matching chunks
from employee_kb (built by ingest.py), and asks OpenAI to answer using ONLY
that retrieved context - grounded, with the source file(s) cited.

Requires an OPENAI_API_KEY set in a .env file in the project root
(intelixsysRAG/.env) - see .env.example for the format. Never commit the
real .env file to git.

Run:
    python chatbot.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # loads OPENAI_API_KEY into os.environ

OPENAI_MODEL = "gpt-4o-mini"   # swap for "gpt-4o" or another chat model if you prefer
TOP_K = 4

SYSTEM_PROMPT = (
    "You are InteliX Systems' internal HR/policy assistant, available only to "
    "verified InteliX Systems employees. Answer questions about internal company "
    "policies (leave, attendance, remote work, benefits, expenses, security, "
    "conduct, onboarding/offboarding, performance reviews, etc.) using ONLY the "
    "CONTEXT provided below - never invent details that aren't in it. "
    "If the answer isn't in the context, say you don't have that information and "
    "suggest the employee contact HR directly. "
    "At the end of your answer, note which policy document(s) the answer came from. "
    "Be clear, concise, and precise, since answers may relate to workplace policy."
)


class EmployeePolicyChatbot:
    def __init__(self, openai_api_key: str | None = None):
        print("Loading embedding model (same one used during ingestion)...")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(
            name="employee_kb", embedding_function=self.embed_fn
        )
        # Falls back to the OPENAI_API_KEY environment variable if not passed in
        self.client = OpenAI(api_key=openai_api_key)

    def retrieve(self, question: str, top_k: int = TOP_K):
        results = self.collection.query(query_texts=[question], n_results=top_k)
        chunks = results["documents"][0] if results["documents"] else []
        sources = [m["source"] for m in results["metadatas"][0]] if results["metadatas"] else []
        return chunks, sources

    def answer(self, question: str, history: list | None = None):
        chunks, sources = self.retrieve(question)
        context = "\n\n---\n\n".join(chunks) if chunks else "(no relevant context found)"

        user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + (history or [])
            + [{"role": "user", "content": user_message}]
        )

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=600,
            messages=messages,
        )

        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "sources": sorted(set(sources)),
        }


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found.")
        print(f"  Create a .env file at {os.path.join(PROJECT_ROOT, '.env')} containing:")
        print('  OPENAI_API_KEY=sk-...')
        print()

    bot = EmployeePolicyChatbot()
    print(f"\nReady - {bot.collection.count()} chunks loaded from employee_kb.")
    print("Ask a question about company policy. Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        result = bot.answer(q)
        print(f"\nBot: {result['answer']}")
        print(f"(sources: {', '.join(result['sources']) if result['sources'] else 'none'})\n")