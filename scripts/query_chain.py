import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ✅ Load environment variables
load_dotenv()

PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main():
    print("🚀 Loading embeddings and vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        PERSIST_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    while True:
        query = input("\n🔍 Ask your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("\n👋 Exiting. Bye!")
            break

        result = qa_chain.invoke({"query": query})

        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

        print("\n✅ Answer:\n")
        print(answer if answer else "⚠️ No answer generated.")

        if sources:
            print("\n📄 Sources:")
            for i, doc in enumerate(sources, 1):
                metadata = doc.metadata or {}
                file_name = metadata.get("file_name", "Unknown file")
                page_number = metadata.get("page_number", "N/A")
                chunk_index = metadata.get("chunk_index", "N/A")
                source_path = metadata.get("source", "Unknown path")

                print(f" {i}. File: {file_name} | Page: {page_number} | Chunk: {chunk_index} | Path: {source_path}")
        else:
            print("\n⚠️ No source documents found.")

if __name__ == "__main__":
    main()
