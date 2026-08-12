import unittest

from fieldguide_ai.config import SessionConfig


class SessionConfigTest(unittest.TestCase):
    def test_session_reconfiguration_returns_a_new_validated_value(self) -> None:
        config = SessionConfig()

        updated = config.with_store(None, None, "documents").with_model("new-model")

        self.assertEqual(config.model, "gpt-5-nano")
        self.assertEqual(config.store_type, "chroma")
        self.assertEqual(updated.model, "new-model")
        self.assertIsNone(updated.store_type)

    def test_session_rejects_partial_store_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a path"):
            SessionConfig(store_type="faiss", store_path=None)
