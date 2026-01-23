from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorstore")


def query_documents(query: str, k: int = 3):
    """
    Retrieve relevant chunks from FAISS and return answer context.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(VECTOR_DB_PATH):
        return {
            "answer": "No documents have been ingested yet.",
            "sources": []
        }

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(query, k=k)

    answer = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))

    return {
        "answer": answer,
        "sources": sources
    }
