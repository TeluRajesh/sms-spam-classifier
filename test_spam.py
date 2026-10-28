"""Unit tests. Run:  python -m unittest discover -s tests -v   (from the project root)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import MAX_CHARS, create_app  # noqa: E402

SPAM = "WINNER!! You have won a free prize. Call 09061701461 now to claim."
HAM = "Hey, are we still meeting for lunch tomorrow?"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")

    def test_spam_message_flagged(self):
        body = self.client.post("/classify", json={"message": SPAM}).get_json()
        self.assertEqual(body["label"], "spam")
        self.assertGreater(body["spam_probability"], 0.5)

    def test_ham_message_not_flagged(self):
        body = self.client.post("/classify", json={"message": HAM}).get_json()
        self.assertEqual(body["label"], "ham")
        self.assertLess(body["spam_probability"], 0.5)

    def test_missing_message_rejected(self):
        self.assertEqual(self.client.post("/classify", json={"text": "hi"}).status_code, 400)

    def test_blank_message_rejected(self):
        self.assertEqual(self.client.post("/classify", json={"message": "   "}).status_code, 400)

    def test_non_string_rejected(self):
        self.assertEqual(self.client.post("/classify", json={"message": 123}).status_code, 400)

    def test_too_long_rejected(self):
        r = self.client.post("/classify", json={"message": "a" * (MAX_CHARS + 1)})
        self.assertEqual(r.status_code, 413)

    def test_non_json_rejected(self):
        r = self.client.post("/classify", data="plain", content_type="text/plain")
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
