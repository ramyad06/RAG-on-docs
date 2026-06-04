"""Document loaders that produce LangChain `Document`s for the ingest pipeline."""
from src.loaders.pdf import load_pdf
from src.loaders.web import load_web_pages

__all__ = ["load_pdf", "load_web_pages"]
