import unittest

from src.config import TOP_K


class RetrievalConfigTest(unittest.TestCase):
    def test_default_retrieval_depth_includes_endpoint_context(self):
        self.assertGreaterEqual(TOP_K, 5)
