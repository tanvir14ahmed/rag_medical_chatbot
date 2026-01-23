from django.shortcuts import render

# Create your views here.
from fastapi import APIRouter
from chatbot.chat.rag_engine import query_documents


router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.get("/query")
def query(question: str):
    """
    Query the RAG system with a question.
    Example: /api/chat/query?question=What+is+the+patient+admission+policy
    """
    result = query_documents(question)
    return result
