import os
import json
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ✅ Absolute paths — bulletproof
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "raw"))
CHUNKS_PATH = os.getenv("CHUNKS_PATH", os.path.join(BASE_DIR, "data", "processed", "chunks.json"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_pdfs():
    print(f"🔍 Looking for PDFs in: {DATA_DIR}")  # This should show the ABS path
    documents = []
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ DATA_DIR does not exist: {DATA_DIR}")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": path,
                                "file_name": file,
                                "page_number": i
                            }
                        })
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc["metadata"].copy()
            chunk_metadata["chunk_index"] = i
            all_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
    return all_chunks

def save_chunks(chunks):
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_PATH}")

def main():
    docs = load_pdfs()
    if not docs:
        print("⚠️  No PDFs found. Exiting.")
        return
    chunks = split_documents(docs)
    save_chunks(chunks)

if __name__ == "__main__":
    main()
