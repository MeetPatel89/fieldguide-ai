import unittest

from fieldguide_ai.messages import ChatMessage


class ChatMessageTest(unittest.TestCase):
    def test_converts_to_openai_input_item(self) -> None:
        message = ChatMessage(role="user", content="Hello")

        self.assertEqual(message.to_input_item(), {"role": "user", "content": "Hello"})
