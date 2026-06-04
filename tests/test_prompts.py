import unittest

from langchain_core.documents import Document

from src.prompts import SYSTEM_PROMPT, build_user_message


class PromptTest(unittest.TestCase):
    def test_required_parameter_questions_must_list_every_required_field(self):
        self.assertIn("required parameters", SYSTEM_PROMPT)
        self.assertIn("list every required parameter", SYSTEM_PROMPT)
        self.assertIn("authorization request vs. access token request", SYSTEM_PROMPT)

    def test_build_user_message_extracts_supported_grant_types(self):
        chunks = [
            Document(
                page_content=(
                    "Supported Grants Authorization Code Grant - requires "
                    "authorization request and access token request calls. "
                    "Implicit Grant - requires an authorization request call. "
                    "Client Credentials Grant - requires an access token request "
                    "call. Refresh Token Grant Type - requires an access token "
                    "request call."
                )
            )
        ]

        message = build_user_message(
            "What OAuth 2.0 grant types does Upwork support?",
            chunks,
        )

        self.assertIn("Extracted supported grant types", message)
        self.assertIn("Authorization Code Grant", message)
        self.assertIn("Implicit Grant", message)
        self.assertIn("Client Credentials Grant", message)
        self.assertIn("Refresh Token Grant Type", message)

    def test_build_user_message_extracts_required_token_parameters(self):
        chunks = [
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
            )
        ]

        message = build_user_message(
            "What parameters are required for the Authorization Code Grant token request?",
            chunks,
        )

        self.assertIn("Extracted required parameters", message)
        self.assertIn("grant_type", message)
        self.assertIn("client_id", message)
        self.assertIn("client_secret", message)
        self.assertIn("code", message)
        self.assertIn("redirect_uri", message)
        self.assertNotIn("Extracted required parameters: response_type", message)

    def test_prompt_treats_unlisted_scope_names_as_gap(self):
        self.assertIn("does not enumerate scope names", SYSTEM_PROMPT)
        self.assertIn("document references scopes but does not list them", SYSTEM_PROMPT)
