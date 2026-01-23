# RAG Medical Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot for medical policies using Django and FastAPI. Users can ask questions and get answers from ingested PDF documents.

## Features

- Ingest PDFs into FAISS vector store
- Use HuggingFace embeddings locally
- Query documents using RAG approach
- FastAPI endpoints for chatting
- Django backend integration

## Setup

```bash
# Clone repo
git clone https://github.com/USERNAME/rag_medical_chatbot.git
cd rag_medical_chatbot

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Ingest PDF documents
python chatbot/chat/ingest.py

# Run FastAPI server
uvicorn chatbot.chatbot.api:app --reload

Test

Root: GET /

Chat: POST /chat/ with JSON:
{
  "question": "What is the hospital policy?"
}


---

# Add `requirements.txt`**

Freeze your dependencies so others can install them:

```bash
pip freeze > requirements.txt
