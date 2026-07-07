import unittest

from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(
        self,
        response_text: str = "Response",
        message_history: list[ChatMessage] | None = None,
    ) -> None:
        super().__init__(message_history=message_history)
        self.response_text = response_text
        self.last_messages: list[ChatMessage] | None = None

    def generate(self, messages: list[ChatMessage]) -> str:
        self.last_messages = list(messages)
        return self.response_text


class ProviderHistoryTest(unittest.TestCase):
    def test_generate_does_not_change_message_history(self) -> None:
        history = [ChatMessage(role="system", content="Be concise.")]
        provider = FakeProvider(message_history=history)

        result = provider.generate([ChatMessage(role="user", content="Hello")])

        self.assertEqual(result, "Response")
        self.assertEqual(provider.get_history(), history)

    def test_chat_appends_user_and_assistant_messages_in_order(self) -> None:
        provider = FakeProvider(
            response_text="Hello back",
            message_history=[ChatMessage(role="system", content="Be helpful.")],
        )

        result = provider.chat("Hello")

        self.assertEqual(result, "Hello back")
        self.assertEqual(
            provider.get_history(),
            [
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hello back"),
            ],
        )
        self.assertEqual(
            provider.last_messages,
            [
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
        )

    def test_chat_accepts_chat_message(self) -> None:
        provider = FakeProvider(response_text="Done")

        provider.chat(ChatMessage(role="user", content="Use this exact message"))

        self.assertEqual(
            provider.get_history(),
            [
                ChatMessage(role="user", content="Use this exact message"),
                ChatMessage(role="assistant", content="Done"),
            ],
        )

    def test_clear_history_removes_existing_messages(self) -> None:
        provider = FakeProvider(message_history=[ChatMessage(role="user", content="Hello")])

        provider.clear_history()

        self.assertEqual(provider.get_history(), [])

    def test_get_history_returns_a_copy(self) -> None:
        provider = FakeProvider(message_history=[ChatMessage(role="user", content="Hello")])

        history = provider.get_history()
        history.append(ChatMessage(role="assistant", content="Mutated externally"))

        self.assertEqual(provider.get_history(), [ChatMessage(role="user", content="Hello")])
