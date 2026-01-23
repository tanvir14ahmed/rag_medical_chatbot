import os
import django
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot.chatbot.settings")
django.setup()


# Import Django apps after django.setup()

from chatbot.chat.views import router as chat_router       # <-- correct import
from chatbot.chat.ingest import ingest_documents           # <-- correct import
from chatbot.chat.rag_engine import query_documents       # <-- correct import


# Create FastAPI app

app = FastAPI(
    title="RAG Medical Chatbot API",
    description="Retrieval-Augmented Generation Chatbot for Medical Policies",
    version="1.0.0",
)


# Optional: Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers from Django apps

app.include_router(chat_router, prefix="/chat", tags=["Chatbot"])


# Root endpoint for testing

@app.get("/")
def root():
    return {"message": "RAG Medical Chatbot API is running!"}
