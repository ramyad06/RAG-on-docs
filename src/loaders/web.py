"""Follow hyperlinks embedded in the source PDF and pull in their text.

The Upwork PDF only includes a partial reference — the load-bearing details
(rate limits, scope tables, team permissions) often live behind hyperlinks to
developers.upwork.com / support.upwork.com. This loader extracts those URLs,
fetches the pages, and returns LangChain `Document`s so they get indexed
alongside the PDF content.

Fetched HTML is cached under `./.web_cache/` so re-ingestion does not hammer
the network.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

ALLOWED_DOMAINS = {
    "developers.upwork.com",
    "www.upwork.com",
    "support.upwork.com",
}

CACHE_DIR = Path(__file__).resolve().parent.parent / ".web_cache"
USER_AGENT = (
    "Mozilla/5.0 (compatible; UpworkAPISupportBot/1.0; "
    "research/educational ingestion)"
)
REQUEST_TIMEOUT = 15
MAX_CHARS_PER_PAGE = 20_000  # safety cap to keep the corpus reasonable


def extract_pdf_urls(pdf_path: Path) -> list[str]:
    """Return unique URLs hyperlinked from the PDF, in first-seen order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for link in (page.hyperlinks or []):
                uri = link.get("uri")
                if not uri:
                    continue
                uri, _ = urldefrag(uri)
                if uri in seen_set:
                    continue
                seen_set.add(uri)
                seen.append(uri)
    return seen


def _is_allowed(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_DOMAINS


def _cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.html"


def _fetch(url: str) -> str | None:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"  ! fetch failed {url}: {exc}")
        return None

    if resp.status_code != 200:
        print(f"  ! HTTP {resp.status_code} {url}")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.text


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Drop noise.
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)

    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_CHARS_PER_PAGE]


def load_web_pages(pdf_path: Path) -> list[Document]:
    """Fetch every allowed-domain link from the PDF and return Documents."""
    urls = [u for u in extract_pdf_urls(pdf_path) if _is_allowed(u)]
    print(f"Following {len(urls)} in-domain links from the PDF...")

    documents: list[Document] = []
    for url in urls:
        html = _fetch(url)
        if not html:
            continue
        text = _html_to_text(html)
        if len(text) < 200:
            # Page rendered to almost nothing (JS-heavy SPA or login wall).
            print(f"  · skipped (too short, {len(text)} chars): {url}")
            continue
        print(f"  · {len(text):>6} chars  {url}")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": url, "kind": "web"},
            )
        )

    return documents
