"""Integration checks for model-runtime messages used by Fieldguide."""

import unittest
from dataclasses import FrozenInstanceError

from model_runtime import Message, MessageRole


class MessageIntegrationTest(unittest.TestCase):
    """Verify Fieldguide's visible turns use normalized runtime messages."""

    def test_stores_normalized_role_and_text(self) -> None:
        message = Message.user("Hello")

        self.assertIs(message.role, MessageRole.USER)
        self.assertEqual(message.text, "Hello")

    def test_rejects_unknown_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported message role"):
            Message(role="critic", content="No")

    def test_is_frozen(self) -> None:
        message = Message.assistant("Hi")

        with self.assertRaises(FrozenInstanceError):
            message.role = MessageRole.USER  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
