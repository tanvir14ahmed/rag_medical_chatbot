import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Base directory of the chatbot project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorstore")
PDF_PATH = os.path.join(BASE_DIR, "policies", "hospital_policy.pdf")  # Correct PDF path

def ingest_documents(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # Local embeddings (no OpenAI API required)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load or create FAISS vectorstore
    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_DB_PATH)

    return {
        "chunks_created": len(chunks),
        "vectorstore_path": VECTOR_DB_PATH
    }


if __name__ == "__main__":
    result = ingest_documents(PDF_PATH)
    print(f"Ingested {result['chunks_created']} chunks from {PDF_PATH}")
    print(f"Vectorstore saved at {result['vectorstore_path']}")
