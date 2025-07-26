import os
import threading
import time
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from symspellpy import SymSpell, Verbosity

# ✅ Load environment variables
load_dotenv()

PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ✅ Custom prompt template
PROMPT_TEMPLATE = """
You are a helpful assistant.
First, try to answer the question using only the provided context.
If the context does not contain relevant information, then answer using your own knowledge,
but clearly say that it is not from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# ✅ Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print(f"⚠️ Failed to load dictionary from {dictionary_path}")

def spell_correct(query: str) -> str:
    """
    Uses SymSpell to correct misspellings in the query.
    If no suggestion is found, returns the original query.
    """
    suggestions = sym_spell.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return query

# ✅ Simulated loading bar
def show_loading(stop_event):
    for i in range(101):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r⏳ Generating answer... {i}%")
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r✅ Response received!           \n")

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

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        query = input("\n🔍 Ask your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("\n👋 Exiting. Bye!")
            break

        corrected_query = spell_correct(query)
        if corrected_query != query:
            print(f"📝 Did you mean: \"{corrected_query}\"")

        # ✅ Start loading animation
        stop_event = threading.Event()
        loader_thread = threading.Thread(target=show_loading, args=(stop_event,))
        loader_thread.start()

        # ✅ Invoke the LLM
        result = qa_chain.invoke({"query": corrected_query})

        # ✅ Stop loading animation
        stop_event.set()
        loader_thread.join()

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
