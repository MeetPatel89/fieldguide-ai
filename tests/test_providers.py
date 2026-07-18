import unittest
from unittest.mock import Mock, patch

from fieldguide_ai.generation import GenerationResult, TokenUsage
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.anthropic_provider import (
    AnthropicProvider,
    from_anthropic_response,
    to_anthropic_messages,
)
from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.providers.openai_provider import (
    OpenAIProvider,
    from_openai_response,
    to_openai_input,
)


class FakeProvider(LLMProvider):
    def __init__(
        self,
        response_text: str = "Response",
        message_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self.response_text = response_text
        self.last_messages: list[ChatMessage] | None = None
        self.last_system_prompt: str | None = None

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        self.last_messages = list(messages)
        self.last_system_prompt = self.system_prompt
        return self._record_generation(
            GenerationResult(
                text=self.response_text,
                provider="fake",
                model="fake-model",
            )
        )


class ProviderHistoryTest(unittest.TestCase):
    def test_generate_does_not_change_message_history(self) -> None:
        history = [ChatMessage(role="user", content="Prior")]
        provider = FakeProvider(
            message_history=history,
            system_prompt="Be concise.",
        )

        result = provider.generate([ChatMessage(role="user", content="Hello")])

        self.assertEqual(result.text, "Response")
        self.assertEqual(provider.get_history(), history)
        self.assertEqual(provider.system_prompt, "Be concise.")

    def test_chat_appends_user_and_assistant_messages_in_order(self) -> None:
        provider = FakeProvider(
            response_text="Hello back",
            system_prompt="Be helpful.",
        )

        result = provider.chat("Hello")

        self.assertEqual(result, "Hello back")
        self.assertEqual(provider.system_prompt, "Be helpful.")
        self.assertEqual(
            provider.get_history(),
            [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hello back"),
            ],
        )
        self.assertEqual(
            provider.last_messages,
            [ChatMessage(role="user", content="Hello")],
        )
        self.assertEqual(provider.last_system_prompt, "Be helpful.")
        self.assertEqual(provider.last_result.text, "Hello back")
        self.assertEqual(len(provider.get_generation_log()), 1)

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

    def test_chat_rejects_non_user_chat_message(self) -> None:
        provider = FakeProvider(response_text="Should not be generated")

        with self.assertRaisesRegex(ValueError, "requires a user-role"):
            provider.chat(ChatMessage(role="assistant", content="Injected turn"))

        self.assertEqual(provider.get_history(), [])
        self.assertEqual(provider.get_generation_log(), [])

    def test_clear_history_keeps_system_prompt_and_generation_log(self) -> None:
        provider = FakeProvider(
            message_history=[ChatMessage(role="user", content="Hello")],
            system_prompt="Stay in character.",
        )
        provider.chat("Next")

        provider.clear_history()

        self.assertEqual(provider.get_history(), [])
        self.assertEqual(provider.system_prompt, "Stay in character.")
        self.assertEqual(len(provider.get_generation_log()), 1)

    def test_generation_log_keeps_defensive_raw_snapshots(self) -> None:
        provider = FakeProvider()
        result = GenerationResult(
            text="Response",
            provider="fake",
            model="fake-model",
            raw={"events": ["created"]},
        )
        provider._record_generation(result)

        assert result.raw is not None
        events = result.raw["events"]
        assert isinstance(events, list)
        events.append("mutated-result")

        logged = provider.last_result
        assert logged is not None
        self.assertEqual(logged.raw, {"events": ["created"]})

        assert logged.raw is not None
        logged_events = logged.raw["events"]
        assert isinstance(logged_events, list)
        logged_events.append("mutated-copy")

        self.assertEqual(
            provider.last_result.raw,
            {"events": ["created"]},
        )

    def test_get_history_returns_a_copy(self) -> None:
        provider = FakeProvider(
            message_history=[ChatMessage(role="user", content="Hello")]
        )

        history = provider.get_history()
        history.append(ChatMessage(role="assistant", content="Mutated externally"))

        self.assertEqual(
            provider.get_history(), [ChatMessage(role="user", content="Hello")]
        )


class MessageAdapterTest(unittest.TestCase):
    def test_to_openai_input_prepends_system_prompt(self) -> None:
        messages = [ChatMessage(role="user", content="Hello")]

        self.assertEqual(
            to_openai_input(messages, system_prompt="Be concise."),
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        )

    def test_to_openai_input_omits_system_when_absent(self) -> None:
        messages = [ChatMessage(role="user", content="Hello")]

        self.assertEqual(
            to_openai_input(messages),
            [{"role": "user", "content": "Hello"}],
        )

    def test_to_anthropic_messages_maps_conversation_turns(self) -> None:
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]

        self.assertEqual(
            to_anthropic_messages(messages),
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )


class GenerationExtractorTest(unittest.TestCase):
    def test_from_openai_response_normalizes_metadata(self) -> None:
        response = Mock(
            output_text="Generated response",
            id="resp_123",
            status="completed",
            usage=Mock(input_tokens=10, output_tokens=4, total_tokens=14),
            model_dump=Mock(return_value={"id": "resp_123"}),
        )

        result = from_openai_response(response, model="test-model", latency_ms=12.5)

        self.assertEqual(
            result,
            GenerationResult(
                text="Generated response",
                provider="openai",
                model="test-model",
                response_id="resp_123",
                finish_reason="completed",
                usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
                latency_ms=12.5,
                raw={"id": "resp_123"},
            ),
        )

    def test_from_anthropic_response_normalizes_metadata(self) -> None:
        response = Mock(
            id="msg_123",
            stop_reason="end_turn",
            content=[Mock(type="text", text="Generated response")],
            usage=Mock(input_tokens=8, output_tokens=3),
            model_dump=Mock(return_value={"id": "msg_123"}),
        )

        result = from_anthropic_response(response, model="test-model", latency_ms=9.0)

        self.assertEqual(
            result,
            GenerationResult(
                text="Generated response",
                provider="anthropic",
                model="test-model",
                response_id="msg_123",
                finish_reason="end_turn",
                usage=TokenUsage(input_tokens=8, output_tokens=3, total_tokens=11),
                latency_ms=9.0,
                raw={"id": "msg_123"},
            ),
        )


class OpenAIProviderTest(unittest.TestCase):
    @patch("fieldguide_ai.providers.openai_provider.OpenAI")
    def test_generate_includes_system_prompt(self, openai_type: Mock) -> None:
        response = Mock(
            output_text="Generated response",
            id="resp_123",
            status="completed",
            usage=None,
            model_dump=Mock(return_value={"id": "resp_123"}),
        )
        openai_type.return_value.responses.create.return_value = response
        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            system_prompt="Be concise.",
        )

        result = provider.generate([ChatMessage(role="user", content="Hello")])

        self.assertEqual(result.text, "Generated response")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.response_id, "resp_123")
        self.assertEqual(provider.last_result, result)
        self.assertEqual(provider.get_generation_log(), [result])
        openai_type.return_value.responses.create.assert_called_once_with(
            model="test-model",
            input=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        )


class AnthropicProviderTest(unittest.TestCase):
    @patch("fieldguide_ai.providers.anthropic_provider.Anthropic")
    def test_generate_passes_system_separately(self, anthropic_type: Mock) -> None:
        text_block = Mock(type="text", text="Generated response")
        response = Mock(
            id="msg_123",
            stop_reason="end_turn",
            content=[text_block],
            usage=Mock(input_tokens=5, output_tokens=2),
            model_dump=Mock(return_value={"id": "msg_123"}),
        )
        anthropic_type.return_value.messages.create.return_value = response
        provider = AnthropicProvider(
            api_key="test-key",
            model="test-model",
            system_prompt="Be concise.",
        )

        result = provider.generate([ChatMessage(role="user", content="Hello")])

        self.assertEqual(result.text, "Generated response")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(provider.last_result, result)
        anthropic_type.return_value.messages.create.assert_called_once_with(
            max_tokens=1000,
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            system="Be concise.",
        )

    @patch("fieldguide_ai.providers.anthropic_provider.Anthropic")
    def test_generate_omits_system_kwarg_when_absent(
        self, anthropic_type: Mock
    ) -> None:
        text_block = Mock(type="text", text="Hi")
        response = Mock(
            id="msg_456",
            stop_reason="end_turn",
            content=[text_block],
            usage=Mock(input_tokens=1, output_tokens=1),
            model_dump=Mock(return_value={}),
        )
        anthropic_type.return_value.messages.create.return_value = response
        provider = AnthropicProvider(api_key="test-key", model="test-model")

        provider.generate([ChatMessage(role="user", content="Hello")])

        anthropic_type.return_value.messages.create.assert_called_once_with(
            max_tokens=1000,
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
