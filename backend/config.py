"""
Configuration settings for the Data Analysis Chatbot.

This file centralizes all the configuration parameters for the application,
making it easy to manage and modify settings without changing the core logic.
"""

import os

# --- General Settings ---
LOG_LEVEL = "INFO"  # Logging level (e.g., DEBUG, INFO, WARNING, ERROR)

# --- File Paths ---
# Base directory for the backend
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for storing uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory for the vector store
VECTOR_STORE_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "vector_store")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- Embedding and Vector Store Settings ---
# Model used for creating text embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index file path
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")

# Metadata file path for the vector store
META_PATH = os.path.join(VECTOR_STORE_DIR, "meta.pkl")

# --- Ollama Language Model Settings ---
# URL of the Ollama server
OLLAMA_BASE_URL = "http://localhost:11434"

# Name of the Ollama model to use for generating responses
OLLAMA_MODEL = "llama3"  # Example: "llama3", "mistral", etc.

# --- Chart Generation Settings ---
# Directory to save generated charts
CHART_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend", "public", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# --- Text Splitting Settings ---
# The size of each text chunk
CHUNK_SIZE = 1000

# --- Server Settings ---
# Base URL for the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# The overlap between consecutive chunks
CHUNK_OVERLAP = 150
