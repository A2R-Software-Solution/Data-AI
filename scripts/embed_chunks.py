import os
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Load env vars
load_dotenv()

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "../data/processed/chunks.json")
PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "../db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    if chunks:
        print(f"🔎 Example metadata: {chunks[0]['metadata']}")
    return chunks

def embed_and_store(chunks):
    print(f"🔗 Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]
    print(f"Created {len(docs)} Document objects with metadata.")

    print(f"Building FAISS vector index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectorstore.save_local(PERSIST_DIR)

    print(f"FAISS index saved to {PERSIST_DIR}")

def main():
    chunks = load_chunks()
    if not chunks:
        print("No chunks found. Exiting.")
        return
    embed_and_store(chunks)

if __name__ == "__main__":
    main()
