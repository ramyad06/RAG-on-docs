import unittest

from src.prompts import SYSTEM_PROMPT


class PromptTest(unittest.TestCase):
    def test_required_parameter_questions_must_list_every_required_field(self):
        self.assertIn("required parameters", SYSTEM_PROMPT)
        self.assertIn("list every required parameter", SYSTEM_PROMPT)
        self.assertIn("authorization request vs. access token request", SYSTEM_PROMPT)
