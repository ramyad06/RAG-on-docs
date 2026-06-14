"""Retrieval layer: open the persisted Chroma index and fetch relevant chunks."""
from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, PDF_PATH, TOP_K

CANDIDATE_K = 24
HTTP_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b")
URL_RE = re.compile(r"https?://\S+")
TIME_VALUE_RE = re.compile(
    r"\b\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
)
REQUIRED_PARAMETER_NAMES = {
    "client_id",
    "client_secret",
    "code",
    "grant_type",
    "redirect_uri",
    "redirect_url",
    "refresh_token",
    "response_type",
}
STOPWORDS = {
    "about",
    "call",
    "code",
    "does",
    "for",
    "from",
    "get",
    "how",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


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


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _query_phrases(query: str) -> set[str]:
    normalized = query.lower()
    phrases = {
        "access token",
        "authorization code",
        "authorization code grant",
        "callback url",
        "client credentials",
        "client secret",
        "rate limit",
        "refresh token",
    }
    return {phrase for phrase in phrases if phrase in normalized}


def _asks_for_endpoint(query: str) -> bool:
    normalized = query.lower()
    endpoint_words = {"endpoint", "url", "uri", "route"}
    if any(word in normalized for word in endpoint_words):
        return True
    return "call" in normalized and any(
        phrase in normalized
        for phrase in ("authorization code", "access token", "refresh token")
    )


def _asks_for_duration(query: str) -> bool:
    normalized = query.lower()
    return any(
        phrase in normalized
        for phrase in (
            "how long",
            "valid for",
            "expires",
            "expire",
            "expiration",
            "ttl",
            "lifetime",
        )
    )


def _asks_for_parameters(query: str) -> bool:
    normalized = query.lower()
    return any(
        phrase in normalized
        for phrase in (
            "parameter",
            "required",
            "requires",
            "need",
            "fields",
        )
    )


def _asks_for_grant_types(query: str) -> bool:
    normalized = query.lower()
    return "grant" in normalized and any(
        phrase in normalized
        for phrase in ("support", "supported", "grant type", "grant types")
    )


def _asks_for_graphql_status(query: str) -> bool:
    normalized = query.lower()
    return "graphql" in normalized and any(
        token in normalized
        for token in ("http", "status", "400", "500", "5xx", "bad request", "scopes")
    )


def _asks_for_access_token_request(query: str) -> bool:
    normalized = query.lower()
    return any(
        phrase in normalized
        for phrase in (
            "access token request",
            "obtain an access token",
            "get an access token",
            "token request",
        )
    )


def _rerank_score(query: str, doc: Document) -> int:
    text = doc.page_content.lower()
    terms = _query_terms(query)
    phrases = _query_phrases(query)

    score = 0
    score += 2 * sum(1 for term in terms if term in text)
    score += 6 * sum(1 for phrase in phrases if phrase in text)

    if _asks_for_endpoint(query):
        if "endpoint" in text:
            score += 10
        if HTTP_METHOD_RE.search(doc.page_content):
            score += 8
        if URL_RE.search(doc.page_content):
            score += 8

    if _asks_for_duration(query):
        if "ttl" in text:
            score += 10
        if "expires_in" in text or "expires in" in text:
            score += 8
        if TIME_VALUE_RE.search(text):
            score += 8

    if _asks_for_parameters(query):
        parameter_hits = sum(1 for name in REQUIRED_PARAMETER_NAMES if name in text)
        score += 5 * parameter_hits
        if "parameters" in text:
            score += 10
        if "required" in text:
            score += 6
        if "grant_type" in text and "authorization_code" in text:
            score += 8
        if "client_secret" in text and "code required" in text:
            score += 8
        if _asks_for_access_token_request(query) and "/oauth2/token" in text:
            score += 12

    if _asks_for_grant_types(query):
        grant_hits = sum(
            1
            for grant in (
                "authorization code grant",
                "implicit grant",
                "client credentials grant",
                "refresh token grant",
            )
            if grant in text
        )
        score += 10 * grant_hits
        if "supported grants" in text:
            score += 20
        if "valid values: code, token" in text or "response_type" in text:
            score -= 10

    if _asks_for_graphql_status(query):
        if "graphql" in text:
            score += 8
        if "200 - ok" in text or "status code 200" in text:
            score += 18
        if "always returns a 200" in text:
            score += 10
        if "400 - bad request" in text and "200" not in text:
            score -= 8

    return score


def _rerank(query: str, docs: list[Document]) -> list[Document]:
    return [
        doc
        for _, doc in sorted(
            enumerate(docs),
            key=lambda item: (-_rerank_score(query, item[1]), item[0]),
        )
    ]


def load_vectorstore() -> Chroma:
    _ensure_vectorstore_exists()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def retrieve(query: str, vectorstore: Chroma, k: int = TOP_K) -> list[Document]:
    candidate_count = max(CANDIDATE_K, k * 3)
    candidates = vectorstore.similarity_search(query, k=candidate_count)
    return _rerank(query, candidates)[:k]


if __name__ == "__main__":
    # Smoke test: confirm retrieval works on a known topic.
    vs = load_vectorstore()
    docs = retrieve("rate limit", vs)
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "?")
        print(f"--- chunk {i} (page {page}) ---")
        print(d.page_content)
        print()
