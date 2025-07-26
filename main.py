import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

# ✅ Prompt template
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

# ✅ Spell checker setup
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print(f"⚠️ Failed to load dictionary from {dictionary_path}")

def spell_correct(query: str) -> str:
    suggestions = sym_spell.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else query

# ✅ Load LLM + Vectorstore
print("🚀 Loading embeddings and vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="mistral")

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ✅ FastAPI setup
app = FastAPI()

# ✅ Allow local frontend to access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/chatAssistant.html")

# ✅ Input model
class QueryInput(BaseModel):
    question: str

# ✅ API route for querying
@app.post("/query")
async def query_chain(input: QueryInput):
    question = input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    corrected = spell_correct(question)
    print(f"🔍 Original: {question}")
    print(f"✅ Corrected: {corrected}")

    result = qa_chain.invoke({"query": corrected})

    answer = result.get("result", "").strip()
    sources = result.get("source_documents", [])

    source_data = []
    for doc in sources:
        metadata = doc.metadata or {}
        source_data.append({
            "file_name": metadata.get("file_name", "Unknown"),
            "page_number": metadata.get("page_number", "N/A"),
            "chunk_index": metadata.get("chunk_index", "N/A"),
            "path": metadata.get("source", "Unknown")
        })

    return {
        "question": question,
        "corrected": corrected,
        "answer": answer or "No answer generated.",
        "sources": source_data
    }
