import os
import json
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load env
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "../data")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "../data/processed/chunks.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

def load_pdfs():
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                file_name = os.path.basename(path)
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            documents.append({
                                "text": text,
                                "metadata": {
                                    "source": path,
                                    "file_name": file_name,
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
            chunk_metadata = {
                "source": doc["metadata"]["source"],
                "file_name": doc["metadata"]["file_name"],
                "page_number": doc["metadata"]["page_number"],
                "chunk_index": i
            }
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
    chunks = split_documents(docs)
    save_chunks(chunks)

if __name__ == "__main__":
    main()
