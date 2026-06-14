import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

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


class _FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.requested_k = None

    def similarity_search(self, query, k):
        self.requested_k = k
        return self.docs[:k]


class RetrieveTest(unittest.TestCase):
    def test_retrieve_fetches_more_candidates_and_promotes_endpoint_context(self):
        docs = [
            Document(
                page_content=(
                    "After redirect, the application gets the authorization code "
                    "from the URL and uses it to request an access token."
                )
            ),
            Document(page_content="Prerequisites include Client ID and Callback URL."),
            Document(page_content="The callback returns code=b094053c2892cb819942."),
            Document(page_content="Access Token Request POST https://example.com/token"),
            Document(page_content="Supported grants include Authorization Code Grant."),
            Document(
                page_content=(
                    "Step 1. Obtaining an authorization code. Endpoint GET "
                    "https://www.upwork.com/ab/account-security/oauth2/authorize "
                    "Parameters response_type required, string. Use code for "
                    "Authorization Code Grant."
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "What endpoint do I call to get an authorization code?",
            vectorstore,
            k=3,
        )

        self.assertGreater(vectorstore.requested_k, 3)
        self.assertIn(
            "https://www.upwork.com/ab/account-security/oauth2/authorize",
            results[0].page_content,
        )
        self.assertEqual(len(results), 3)

    def test_retrieve_promotes_chunks_with_matching_query_terms(self):
        docs = [
            Document(page_content="General Upwork legal terms and platform overview."),
            Document(page_content="Authentication uses OAuth 2.0 grants."),
            Document(page_content="The access token TTL is 24 hours."),
            Document(
                page_content=(
                    "Rate limits are 300 requests per minute per IP and "
                    "40K requests per day."
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "What is the rate limit per IP?",
            vectorstore,
            k=2,
        )

        self.assertGreater(vectorstore.requested_k, 2)
        self.assertIn("300 requests per minute per IP", results[0].page_content)
        self.assertEqual(len(results), 2)

    def test_retrieve_promotes_duration_context_for_validity_questions(self):
        docs = [
            Document(page_content="Endpoint POST https://www.upwork.com/api/v3/oauth2/token"),
            Document(page_content="A valid refresh token can obtain a new access token."),
            Document(page_content="The access token request returns token fields."),
            Document(
                page_content=(
                    "TTL for an access token is 24 hours; TTL for a refresh "
                    "token is 2 weeks since its last usage."
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "How long is an OAuth access token valid for?",
            vectorstore,
            k=2,
        )

        self.assertIn("24 hours", results[0].page_content)
        self.assertEqual(len(results), 2)

    def test_retrieve_promotes_required_parameter_tables(self):
        docs = [
            Document(
                page_content=(
                    "Authorization Code Grant is an OAuth 2.0 flow that enables "
                    "you to obtain an access token."
                )
            ),
            Document(
                page_content=(
                    "Prerequisites Data Description Client ID For each application "
                    "you develop, you must obtain a client identifier key."
                )
            ),
            Document(
                page_content=(
                    "Endpoint GET https://www.upwork.com/ab/account-security/oauth2/"
                    "authorize Parameters response_type required, string client_id "
                    "required, string redirect_uri required, string"
                )
            ),
            Document(
                page_content=(
                    "Endpoint POST https://www.upwork.com/api/v3/oauth2/token "
                    "Parameters grant_type required, string client_id required, "
                    "string client_secret required, string code required, string "
                    "redirect_uri required, string"
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "What are the required parameters for the Authorization Code Grant "
            "access token request?",
            vectorstore,
            k=2,
        )

        self.assertIn("grant_type required", results[0].page_content)
        self.assertIn("client_secret required", results[0].page_content)
        self.assertIn("redirect_uri required", results[0].page_content)

    def test_retrieve_promotes_supported_grants_section(self):
        docs = [
            Document(
                page_content=(
                    "Endpoint GET https://www.upwork.com/ab/account-security/"
                    "oauth2/authorize Parameters response_type required, string. "
                    "Valid values: code, token."
                )
            ),
            Document(
                page_content=(
                    "Supported Grants Authorization Code Grant - requires "
                    "authorization request and access token request calls. "
                    "Implicit Grant - requires an authorization request call. "
                    "Client Credentials Grant - requires an access token request "
                    "call. Refresh Token Grant Type - requires an access token "
                    "request call."
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "What OAuth 2.0 grant types does Upwork support?",
            vectorstore,
            k=1,
        )

        self.assertIn("Supported Grants", results[0].page_content)
        self.assertIn("Client Credentials Grant", results[0].page_content)

    def test_retrieve_promotes_graphql_status_code_explanation(self):
        docs = [
            Document(
                page_content=(
                    "REST endpoints return HTTP status codes in the error "
                    "response, for example, 400 - Bad Request."
                )
            ),
            Document(
                page_content=(
                    "In case of malformed syntax, GraphQL always returns a "
                    "200 - OK status code, regardless of whether the operation "
                    "succeeded or failed."
                )
            ),
        ]
        vectorstore = _FakeVectorstore(docs)

        results = rag.retrieve(
            "Does Upwork's GraphQL API return HTTP 400 for bad requests?",
            vectorstore,
            k=1,
        )

        self.assertIn("200 - OK", results[0].page_content)
