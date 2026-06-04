"""Retrieval layer: open the persisted Chroma index and fetch top-k chunks."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, PDF_PATH, TOP_K


def _ensure_vectorstore_exists() -> None:
    if CHROMA_DIR.exists():
        return

    if PDF_PATH.exists():
        from src.ingest import main as build_index

        build_index()

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run `python -m src.ingest` first, or include the source PDF so "
            "the app can build the index on first startup."
        )


def load_vectorstore() -> Chroma:
    _ensure_vectorstore_exists()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def retrieve(query: str, vectorstore: Chroma, k: int = TOP_K) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    # Smoke test: confirm retrieval works on a known topic.
    vs = load_vectorstore()
    docs = retrieve("rate limit", vs)
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "?")
        print(f"--- chunk {i} (page {page}) ---")
        print(d.page_content)
        print()
