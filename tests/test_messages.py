import unittest

from fieldguide_ai.chat import ChatMessage


class ChatMessageTest(unittest.TestCase):
    def test_stores_role_and_content(self) -> None:
        message = ChatMessage(role="user", content="Hello")

        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello")

    def test_is_immutable(self) -> None:
        message = ChatMessage(role="assistant", content="Hi")

        with self.assertRaises(Exception):
            message.content = "changed"  # type: ignore[misc]
