"""
Highlighter Module - Phase 3
PDF page highlighting with high-resolution image rendering

Opens PDFs, highlights specific text, and returns crisp page images.
"""

from pathlib import Path
from typing import List
import fitz  # PyMuPDF


# Configuration
PDF_DIRECTORY = Path("./data/pdfs")


def find_text_with_fallback(page: fitz.Page, text: str) -> List[fitz.Rect]:
    """
    Search for text on a page with fallback logic for partial matches.

    Args:
        page: PyMuPDF page object
        text: Text to search for

    Returns:
        List of rectangles where text was found
    """
    # Primary search - exact match
    rects = page.search_for(text)

    if rects:
        return rects

    # Fallback 1: Try searching for the first sentence
    sentences = text.split(". ")
    if sentences and sentences[0]:
        rects = page.search_for(sentences[0])
        if rects:
            return rects

    # Fallback 2: Try overlapping 5-word phrases
    words = text.split()
    if len(words) >= 5:
        for i in range(len(words) - 4):
            phrase = " ".join(words[i : i + 5])
            rects = page.search_for(phrase)
            if rects:
                return rects

    # Fallback 3: Try individual sentences
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # Only search substantial sentences
            rects = page.search_for(sentence.strip())
            if rects:
                return rects

    # Fallback 4: Try first 50 characters
    if len(text) > 50:
        rects = page.search_for(text[:50])
        if rects:
            return rects

    return []


def get_highlighted_page_image(
    pdf_filename: str, page_num: int, text_to_highlight: str
) -> bytes:
    """
    Open a PDF, highlight specific text on a page, and return a high-res image.

    Args:
        pdf_filename: Name of the PDF file (e.g., "marx_capital.pdf")
        page_num: 1-indexed page number (human-readable)
        text_to_highlight: Text snippet to highlight on the page

    Returns:
        PNG image bytes of the highlighted page

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If page number is invalid or text not found
    """
    # Build full path
    pdf_path = PDF_DIRECTORY / pdf_filename

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open document with context manager
    with fitz.open(pdf_path) as doc:
        # Validate page number (convert from 1-indexed to 0-indexed)
        page_idx = page_num - 1

        if page_idx < 0 or page_idx >= len(doc):
            raise ValueError(
                f"Invalid page number {page_num}. Document has {len(doc)} pages."
            )

        # Load the page
        page = doc.load_page(page_idx)

        # Search for text with fallback logic
        rects = find_text_with_fallback(page, text_to_highlight)

        if not rects:
            raise ValueError(
                f"Text not found on page {page_num}: '{text_to_highlight[:50]}...'"
            )

        # Apply highlights
        for rect in rects:
            page.add_highlight_annot(rect)

        # Render at high resolution (2x zoom)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        image_bytes = pix.tobytes("png")

        return image_bytes


if __name__ == "__main__":
    # Testing Gate
    print("=" * 70)
    print("HIGHLIGHTER MODULE - TESTING GATE")
    print("=" * 70)

    # Test configuration - adjust these based on your actual PDFs
    test_pdf = "SOCIALISM UTOPIAN SCIENTIFIC.pdf"  # Change this to an actual PDF in ./data/pdfs/
    test_page = 1
    test_text = "The mode of production"  # Change this to actual text from the PDF

    print(f"\nTest Configuration:")
    print(f"  PDF: {test_pdf}")
    print(f"  Page: {test_page}")
    print(f"  Text: '{test_text[:50]}...'")
    print("-" * 70)

    try:
        # Generate highlighted image
        image_bytes = get_highlighted_page_image(test_pdf, test_page, test_text)

        # Write to test file
        output_path = Path("test_highlight.png")
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"\n✓ Successfully generated highlighted image")
        print(f"✓ Saved to: {output_path.absolute()}")
        print(f"✓ Image size: {len(image_bytes)} bytes")
        print("\n" + "=" * 70)
        print("✅ TESTING GATE COMPLETE - Check test_highlight.png")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nPlease ensure:")
        print("  1. The PDF exists in ./data/pdfs/")
        print("  2. Update test_pdf variable with actual filename")
        print("  3. Update test_text with actual text from the PDF")

    except ValueError as e:
        print(f"\n❌ Search error: {e}")
        print("\nThe text was not found. Try:")
        print("  1. Checking the exact text on the page")
        print("  2. Using a shorter, unique phrase from the page")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
