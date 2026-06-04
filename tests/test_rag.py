import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src import rag


class LoadVectorstoreTest(unittest.TestCase):
    def test_load_vectorstore_builds_index_when_pdf_exists(self):
        with self.subTest("missing Chroma index and present PDF"):
            self._assert_load_vectorstore_builds_index_when_pdf_exists()

    def _assert_load_vectorstore_builds_index_when_pdf_exists(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chroma_dir = tmp_path / "chroma_db"
            pdf_path = tmp_path / "docs" / "API Documentation Partial.pdf"
            pdf_path.parent.mkdir()
            pdf_path.write_bytes(b"%PDF-1.4\n")

            def fake_ingest_main():
                chroma_dir.mkdir()

            fake_ingest = types.SimpleNamespace(main=fake_ingest_main)

            with patch.object(rag, "CHROMA_DIR", chroma_dir), patch.object(
                rag, "PDF_PATH", pdf_path
            ), patch.dict(sys.modules, {"src.ingest": fake_ingest}), patch.object(
                rag, "HuggingFaceEmbeddings", return_value="embeddings"
            ), patch.object(rag, "Chroma", return_value="vectorstore") as chroma:
                result = rag.load_vectorstore()

            self.assertEqual(result, "vectorstore")
            chroma.assert_called_once_with(
                collection_name=rag.COLLECTION_NAME,
                persist_directory=str(chroma_dir),
                embedding_function="embeddings",
            )
