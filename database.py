"""
Database Module
Handles ChromaDB initialization and document storage with local embeddings
"""

import shutil
from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# Configuration
CHROMA_DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "pdf_library"
EMBEDDING_API_BASE = "http://100.72.139.71:11434"
EMBEDDING_MODEL = "embeddinggemma:latest"


def get_embeddings() -> OllamaEmbeddings:
    """
    Initialize Ollama embeddings for local Gemma model.

    Returns:
        OllamaEmbeddings instance configured for local Ollama/Gemma
    """
    return OllamaEmbeddings(
        base_url=EMBEDDING_API_BASE,
        model=EMBEDDING_MODEL,
    )


def reset_chroma_db() -> None:
    """
    Clear the ChromaDB directory to ensure idempotency.
    Removes all existing data to prevent duplicate chunks.
    """
    if CHROMA_DB_DIR.exists():
        print(f"Clearing existing ChromaDB at {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


def initialize_vectorstore(documents: Optional[List[Document]] = None) -> Chroma:
    """
    Initialize ChromaDB vector store.

    Args:
        documents: Optional list of documents to add immediately

    Returns:
        Chroma vector store instance
    """
    embeddings = get_embeddings()

    if documents:
        # Create with documents
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DB_DIR),
        )
    else:
        # Create empty vectorstore
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DB_DIR),
        )

    return vectorstore


def store_documents(documents: List[Document]) -> Chroma:
    """
    Store documents in ChromaDB with embeddings.

    Args:
        documents: List of LangChain Document objects to store

    Returns:
        Chroma vector store instance
    """
    # Reset for idempotency
    reset_chroma_db()

    print(f"Storing {len(documents)} chunks in ChromaDB...")
    vectorstore = initialize_vectorstore(documents=documents)
    print(f"Successfully stored documents in collection '{COLLECTION_NAME}'")

    return vectorstore


def query_vectorstore(vectorstore: Chroma, query: str, k: int = 3) -> List[Document]:
    """
    Query the vector store for similar documents.

    Args:
        vectorstore: Chroma vector store instance
        query: Search query string
        k: Number of results to return (default: 3)

    Returns:
        List of matching Document objects
    """
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    # Test the database setup
    from extractor import extract_all_pdfs
    from chunker import chunk_documents

    pdf_dir = Path("./data/pdfs")

    print("Step 1: Extracting PDFs...")
    docs = extract_all_pdfs(pdf_dir)

    if docs:
        print("\nStep 2: Chunking documents...")
        chunked = chunk_documents(docs)

        print("\nStep 3: Storing in ChromaDB...")
        vectorstore = store_documents(chunked)

        print("\nStep 4: Testing query...")
        results = query_vectorstore(vectorstore, "test query", k=3)

        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Metadata: {doc.metadata}")
            print(f"Content: {doc.page_content[:200]}...")
    else:
        print("No documents to process. Add PDFs to ./data/pdfs/")
