"""
PDF Text Extraction Module
Extracts text from PDF files using PyMuPDF (fitz)
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a single PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries with schema:
        {"text": "extracted string", "metadata": {"page_number": int, "source": "filename.pdf"}}

    Note:
        PyMuPDF uses 0-indexed page numbers, but we convert to 1-indexed for human readability.
    """
    documents = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text()
            # Handle different return types from get_text()
            if isinstance(raw_text, (list, dict)):
                continue
            text = raw_text

            # Skip empty pages
            if not text.strip():
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "page_number": page_num + 1,  # Convert to 1-indexed
                        "source": pdf_path.name,
                    },
                }
            )

    return documents


def extract_all_pdfs(pdf_directory: Path) -> List[Dict[str, Any]]:
    """
    Extract text from all PDF files in a directory.

    Args:
        pdf_directory: Path to directory containing PDF files

    Returns:
        List of all extracted documents from all PDFs
    """
    all_documents = []

    if not pdf_directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

    pdf_files = list(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        print(f"Warning: No PDF files found in {pdf_directory}")
        return all_documents

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        documents = extract_text_from_pdf(pdf_file)
        all_documents.extend(documents)
        print(f"  Extracted {len(documents)} pages")

    return all_documents


if __name__ == "__main__":
    # Test the extraction
    pdf_dir = Path("./data/pdfs")
    docs = extract_all_pdfs(pdf_dir)
    print(f"\nTotal documents extracted: {len(docs)}")
    if docs:
        print(f"Sample: {docs[0]['metadata']}")
