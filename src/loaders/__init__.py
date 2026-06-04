"""Document loaders that produce LangChain `Document`s for the ingest pipeline."""
from src.loaders.pdf import load_pdf

__all__ = ["load_pdf"]
