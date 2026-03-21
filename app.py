"""
Semantic PDF Highlighter - Phase 4
Streamlit web interface for RAG-powered PDF search and highlighting

Features:
- Search for concepts across PDF library
- View cited text snippets with automatic highlighting
- Paginated results for better UX
- Cached database connection for performance
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from retriever import search_library
from highlighter import get_highlighted_page_image


# Configuration - must match Phase 1
CHROMA_DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "pdf_library"
EMBEDDING_API_BASE = "http://100.72.139.71:11434"
EMBEDDING_MODEL = "embeddinggemma:latest"


@st.cache_resource
def get_cached_vectorstore() -> Chroma:
    """
    Initialize and cache the ChromaDB connection.
    This is only called once per session for performance.
    """
    embeddings = OllamaEmbeddings(
        base_url=EMBEDDING_API_BASE,
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vectorstore


# Initialize session state to persist data across reruns
if "results" not in st.session_state:
    st.session_state.results = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "images_cache" not in st.session_state:
    st.session_state.images_cache = {}  # Cache for highlighted images


def render_result(result: Dict[str, Any], idx: int) -> None:
    """
    Render a single search result with automatic image generation.

    Args:
        result: Dictionary containing text, source, page_number, citation
        idx: Result index for caching
    """
    with st.container():
        # Display text snippet
        st.write(f'"{result["text"]}"')

        # Display citation
        st.caption(result["citation"])

        # Check if image is already cached
        cache_key = f"{result['source']}_{result['page_number']}_{idx}"

        if cache_key not in st.session_state.images_cache:
            # Generate highlighted image
            with st.spinner("Generating highlight..."):
                try:
                    image_bytes = get_highlighted_page_image(
                        pdf_filename=result["source"],
                        page_num=result["page_number"],
                        text_to_highlight=result["text"],
                    )
                    st.session_state.images_cache[cache_key] = image_bytes
                except Exception as e:
                    st.error(f"Could not generate highlight: {e}")
                    return

        # Display the highlighted image
        if cache_key in st.session_state.images_cache:
            st.image(
                st.session_state.images_cache[cache_key],
                width="content",  # Use container width
            )

            # Download button
            st.download_button(
                label="💾 Download",
                data=st.session_state.images_cache[cache_key],
                file_name=f"highlight_{result['source'].replace('.pdf', '').replace(' ', '_')}_p{result['page_number']}.png",
                mime="image/png",
                key=f"dl_{cache_key}",
            )

        st.divider()


def main():
    """Main Streamlit application."""
    # Initialize cached vectorstore (only runs once)
    _ = get_cached_vectorstore()

    # Header
    st.title("📚 Semantic PDF Highlighter")
    st.markdown("Search your PDF library and view highlighted source pages.")
    st.divider()

    # Search Section
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., Engel's ideas on machinery",
        key="search_input",
    )

    col_search, col_spacer = st.columns([1, 4])
    with col_search:
        search_clicked = st.button("🔍 Search", key="search_btn")

    # Handle search
    if search_clicked and query:
        with st.spinner("Searching..."):
            try:
                # Use cached vectorstore for better performance
                cached_vectorstore = get_cached_vectorstore()
                st.session_state.results = search_library(
                    query, top_k=20, vectorstore=cached_vectorstore
                )
                st.session_state.current_page = 0
                st.session_state.images_cache = {}  # Clear image cache

                if not st.session_state.results:
                    st.warning("No results found. Try a different query.")
            except Exception as e:
                st.error(f"Search error: {e}")
                st.info("Make sure Phase 1 has been run and ChromaDB is populated.")

    st.divider()

    # Results section
    if st.session_state.results:
        st.subheader(f"Search Results ({len(st.session_state.results)} found)")

        # Pagination controls
        RESULTS_PER_PAGE = 3
        total_results = len(st.session_state.results)
        total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE

        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()

        with col_info:
            st.caption(f"Page {st.session_state.current_page + 1} of {total_pages}")

        with col_next:
            if st.button(
                "Next ➡️", disabled=st.session_state.current_page >= total_pages - 1
            ):
                st.session_state.current_page += 1
                st.rerun()

        # Calculate slice for current page
        start_idx = st.session_state.current_page * RESULTS_PER_PAGE
        end_idx = min(start_idx + RESULTS_PER_PAGE, total_results)

        # Display results for current page
        for idx in range(start_idx, end_idx):
            result = st.session_state.results[idx]
            render_result(result, idx)

        # Bottom pagination controls
        st.divider()
        col_prev2, col_info2, col_next2 = st.columns([1, 2, 1])

        with col_prev2:
            if st.button(
                "⬅️ Previous", disabled=st.session_state.current_page == 0, key="prev2"
            ):
                st.session_state.current_page -= 1
                st.rerun()

        with col_info2:
            st.caption(f"Showing results {start_idx + 1}-{end_idx} of {total_results}")

        with col_next2:
            if st.button(
                "Next ➡️",
                disabled=st.session_state.current_page >= total_pages - 1,
                key="next2",
            ):
                st.session_state.current_page += 1
                st.rerun()
    else:
        st.info("Enter a query and click Search to find relevant passages.")

        # Placeholder illustration
        st.markdown("""
        ### How it works:
        1. **Search**: Enter a concept or question in the search bar
        2. **Browse**: Review the semantic search results
        3. **View**: Each result automatically shows the highlighted PDF page
        4. **Navigate**: Use pagination to browse through all results
        5. **Download**: Save highlighted images for reference
        """)


if __name__ == "__main__":
    main()
