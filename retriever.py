"""
Retriever Module - Phase 2
Search interface for the RAG Semantic Highlighter application

Connects to existing ChromaDB and provides citation-ready search results.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Configuration - must match Phase 1
CHROMA_DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "pdf_library"
EMBEDDING_API_BASE = "http://100.72.139.71:11434"
EMBEDDING_MODEL = "embeddinggemma:latest"


def get_embeddings() -> OllamaEmbeddings:
    """
    Initialize Ollama embeddings - same configuration as Phase 1.

    Returns:
        OllamaEmbeddings instance configured for local Gemma model
    """
    return OllamaEmbeddings(
        base_url=EMBEDDING_API_BASE,
        model=EMBEDDING_MODEL,
    )


def load_vectorstore() -> Chroma:
    """
    Load existing ChromaDB in read-only/query mode.

    Returns:
        Chroma vector store instance connected to existing database
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vectorstore


def search_library(
    query: str, top_k: int = 5, vectorstore: Chroma = None
) -> List[Dict[str, Any]]:
    """
    Search the PDF library for relevant chunks.

    Args:
        query: User search query string
        top_k: Number of results to return (default: 5)
        vectorstore: Optional pre-initialized Chroma vectorstore for caching

    Returns:
        List of dictionaries with schema:
        {
            "text": "The extracted chunk text...",
            "source": "filename.pdf",
            "page_number": 42,
            "citation": "(filename.pdf, p. 42)"
        }
    """
    # Load the vector store (read-only) if not provided
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=top_k)

    # Format results according to strict schema
    formatted_results = []

    for doc in results:
        # Extract metadata
        source = doc.metadata.get("source", "unknown")
        page_number = doc.metadata.get("page_number", 0)

        # Generate citation
        citation = f"({source}, p. {page_number})"

        # Build formatted result
        formatted_result = {
            "text": doc.page_content,
            "source": source,
            "page_number": page_number,
            "citation": citation,
        }

        formatted_results.append(formatted_result)

    return formatted_results


if __name__ == "__main__":
    # Testing Gate
    from pprint import pprint

    print("=" * 70)
    print("RETRIEVER MODULE - TESTING GATE")
    print("=" * 70)
    print(f"\nQuery: Engel's ideas on machinery")
    print(f"Top K: 3")
    print("-" * 70)

    try:
        results = search_library("Engel's ideas on machinery", top_k=3)

        print(f"\n✓ Retrieved {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"--- Result {i} ---")
            pprint(result)
            print()

        print("=" * 70)
        print("✅ TESTING GATE COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        print("\nMake sure Phase 1 has been run and ChromaDB is populated.")
        raise
