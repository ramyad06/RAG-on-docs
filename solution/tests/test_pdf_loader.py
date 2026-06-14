import unittest
from pathlib import Path
from unittest.mock import patch

from src.loaders.pdf import load_pdf


class _FakePdf:
    pages = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePage:
    def extract_text(self):
        return "OAuth access tokens are valid for 24 hours."

    def extract_tables(self):
        return []


class PdfLoaderTest(unittest.TestCase):
    def test_pdf_source_metadata_is_repo_relative(self):
        fake_pdf = _FakePdf()
        fake_pdf.pages = [_FakePage()]
        pdf_path = Path("/tmp/project/docs/API Documentation Partial.pdf")

        with patch.object(Path, "exists", return_value=True), patch(
            "src.loaders.pdf.pdfplumber.open", return_value=fake_pdf
        ):
            docs = load_pdf(pdf_path)

        self.assertEqual(docs[0].metadata["source"], "docs/API Documentation Partial.pdf")
