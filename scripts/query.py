import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Load env vars
load_dotenv()

PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

    query = input("🔍 Enter your query: ")
    results = vectorstore.similarity_search(query, k=3)

    print("\n✅ Top results:")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()
