"""One-time ingestion: PDF -> chunks -> embeddings -> Chroma.

Run from the project root:  python -m src.ingest
"""
from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATH,
)
from src.loaders import load_pdf


def sanity_check(pages: list[Document]) -> None:
    """Assignment A1: print total char count and a sample of the text."""
    if not pages:
        raise ValueError(
            f"Loader returned 0 pages from {PDF_PATH}. "
            "The PDF may be empty, image-only, or corrupt."
        )
    total_chars = sum(len(p.page_content) for p in pages)
    print(f"Loaded {len(pages)} pages from {PDF_PATH.name}")
    print(f"Total characters: {total_chars:,}")
    sample = pages[0].page_content[:500].replace("\n", " ")
    print(f"\nSample (page 1, first 500 chars):\n{sample}\n")


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]) -> None:
    # Wipe any prior index so re-runs don't double-insert.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def main() -> None:
    pages = load_pdf(PDF_PATH)
    sanity_check(pages)

    chunks = chunk_documents(pages)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("Embedding and writing to Chroma (this may take a minute on first run)...")
    build_vectorstore(chunks)
    print(f"Done. Index saved at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
