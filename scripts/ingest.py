import os
import json
from pathlib import Path
from dotenv import load_dotenv

from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings

# Load env vars
load_dotenv()

# Fixed configurations - using environment variables only
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./data/processed/chunks.json")  # Fixed typo and path
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MONGO_URI = os.getenv("MONGO_URI")  # NEVER hardcode credentials
MONGO_DB = os.getenv("MONGO_DB", "rag_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")

def load_chunks():
    """Load chunks from JSON file with proper error handling."""
    chunks_file = Path(CHUNKS_PATH)
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
        return chunks
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in chunks file: {e}")

def embed_and_store(chunks):
    """Embed chunks and store in MongoDB Atlas with batch processing."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set")
    
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    try:
        client = MongoClient(MONGO_URI)
        # Test connection
        client.admin.command('ping')
        
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        print(f"Connected to MongoDB Atlas: {MONGO_DB}.{MONGO_COLLECTION}")

        # Prepare documents with embeddings (batch processing)
        documents = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            embedding = embeddings.embed_query(chunk["text"])
            doc = {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            }
            documents.append(doc)

        # Batch insert for better performance
        if documents:
            result = collection.insert_many(documents, ordered=False)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} chunks with embeddings into MongoDB.")
        else:
            print("No documents to insert.")
            
    except Exception as e:
        print(f"Error connecting to MongoDB or inserting documents: {e}")
        raise

def main():
    """Main function with proper error handling."""
    try:
        chunks = load_chunks()
        if not chunks:
            print("No chunks found. Exiting.")
            return
        
        embed_and_store(chunks)
        print("✅ Ingestion completed successfully!")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())