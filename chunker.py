"""
Text Chunking Module
Chunks extracted text using LangChain's RecursiveCharacterTextSplitter
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata.

    Args:
        documents: List of dicts with schema {"text": str, "metadata": dict}
        chunk_size: Size of each chunk in characters (default: 1000)
        chunk_overlap: Overlap between chunks in characters (default: 200)

    Returns:
        List of LangChain Document objects with preserved metadata
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Convert to LangChain Document objects
    lc_documents = [
        Document(page_content=doc["text"], metadata=doc["metadata"])
        for doc in documents
    ]

    # Split documents into chunks
    chunked_docs = text_splitter.split_documents(lc_documents)

    return chunked_docs


if __name__ == "__main__":
    # Test the chunking
    from extractor import extract_all_pdfs
    from pathlib import Path

    pdf_dir = Path("./data/pdfs")
    docs = extract_all_pdfs(pdf_dir)

    if docs:
        chunked = chunk_documents(docs)
        print(f"\nOriginal documents: {len(docs)}")
        print(f"Chunked documents: {len(chunked)}")
        print(f"\nSample chunk metadata: {chunked[0].metadata}")
        print(
            f"Sample chunk content (first 200 chars): {chunked[0].page_content[:200]}..."
        )
