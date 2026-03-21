#!/usr/bin/env python3
"""
RAG Semantic Highlighter - Phase 1
Main ingestion script for PDF documents

Pipeline:
1. Extract text from PDFs using PyMuPDF
2. Chunk text using LangChain RecursiveCharacterTextSplitter
3. Embed chunks using local Gemma model via OpenAI-compatible API
4. Store in ChromaDB
5. Run testing gate with dummy query
"""

from pathlib import Path
from extractor import extract_all_pdfs
from chunker import chunk_documents
from database import store_documents, query_vectorstore


def main():
    """Main ingestion pipeline with testing gate."""
    pdf_dir = Path("./data/pdfs")

    print("=" * 60)
    print("RAG SEMANTIC HIGHLIGHTER - PHASE 1")
    print("=" * 60)

    # Step 1: Extract text from PDFs
    print("\n[1/4] Extracting text from PDFs...")
    print("-" * 40)
    documents = extract_all_pdfs(pdf_dir)

    if not documents:
        print("\n⚠️  No PDFs found in ./data/pdfs/")
        print("Please add PDF files and try again.")
        return

    print(f"✓ Extracted {len(documents)} pages total")

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    print("-" * 40)
    chunked_docs = chunk_documents(documents)
    print(f"✓ Created {len(chunked_docs)} chunks")
    print(f"  (chunk_size=1000, chunk_overlap=200)")

    # Step 3: Store in ChromaDB
    print("\n[3/4] Storing in ChromaDB...")
    print("-" * 40)
    vectorstore = store_documents(chunked_docs)
    print(f"✓ Stored in collection 'pdf_library'")
    print(f"✓ Database location: ./chroma_db/")

    # Step 4: Testing Gate
    print("\n[4/4] Testing Gate - Query Test")
    print("=" * 60)
    print("Running dummy query: 'test query'")
    print("-" * 40)

    results = query_vectorstore(vectorstore, "test query", k=3)

    print(f"\n✓ Retrieved {len(results)} results:\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...")
        print()

    print("=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
