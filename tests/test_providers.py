import unittest
from collections.abc import AsyncIterator, Callable, Sequence

from model_runtime import (
    ChatModel,
    FinishReason,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ProviderUnavailableError,
    StreamEvent,
    Usage,
)

from fieldguide_ai.chat import ChatMessage, GenerationResult, TokenUsage
from fieldguide_ai.errors import ConfigurationError, ProviderGenerationError
from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.providers.runtime_provider import ModelRuntimeProvider


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

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
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
        class RawProvider(FakeProvider):
            def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
                return self._record_generation(
                    GenerationResult(
                        text="Response",
                        provider="fake",
                        model="fake-model",
                        raw={"events": ["created"]},
                    )
                )

        provider = RawProvider()
        result = provider.generate([])

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

    def test_failed_chat_does_not_leave_a_partial_turn_in_history(self) -> None:
        class FailingProvider(FakeProvider):
            def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
                raise RuntimeError("generation failed")

        provider = FailingProvider(
            message_history=[ChatMessage(role="user", content="Prior")]
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            provider.chat("New question")

        self.assertEqual(
            provider.get_history(),
            [ChatMessage(role="user", content="Prior")],
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


class FakeChatModel(ChatModel):
    def __init__(
        self,
        response: ModelResponse | None = None,
        error: ProviderUnavailableError | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.requests: list[tuple[str, ModelRequest]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(streaming=False)

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        self.requests.append((model_id, request))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield


def build_runtime_provider(
    adapter: ChatModel,
    *,
    system_prompt: str | None = None,
    clock: Callable[[], float] | None = None,
) -> ModelRuntimeProvider:
    runtime = ModelRuntime(ModelRouter({"primary": (adapter, "test-model")}))
    return ModelRuntimeProvider(
        runtime=runtime,
        logical_name="primary",
        provider_name="fake",
        model="test-model",
        system_prompt=system_prompt,
        clock=clock,
    )


class ModelRuntimeProviderTest(unittest.TestCase):
    def test_generate_maps_request_response_and_records_result(self) -> None:
        response = ModelResponse(
            message=Message.assistant("Generated response"),
            usage=Usage(input_tokens=10, output_tokens=4),
            finish_reason=FinishReason.STOP,
            raw={"id": "resp_123", "kind": "response"},
            model="returned-model",
        )
        adapter = FakeChatModel(response=response)
        ticks = iter((0.0, 0.0125))
        provider = build_runtime_provider(
            adapter,
            system_prompt="Be concise.",
            clock=lambda: next(ticks),
        )

        result = provider.generate(
            [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi"),
            ]
        )

        self.assertEqual(
            result,
            GenerationResult(
                text="Generated response",
                provider="fake",
                model="returned-model",
                response_id="resp_123",
                finish_reason="stop",
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                ),
                latency_ms=12.5,
                raw={"id": "resp_123", "kind": "response"},
            ),
        )
        self.assertEqual(provider.last_result, result)
        self.assertEqual(provider.get_generation_log(), [result])
        model_id, request = adapter.requests[0]
        self.assertEqual(model_id, "test-model")
        self.assertEqual(
            request.messages,
            (
                Message.system("Be concise."),
                Message.user("Hello"),
                Message.assistant("Hi"),
            ),
        )

    def test_generate_wraps_runtime_errors_and_preserves_context(self) -> None:
        runtime_error = ProviderUnavailableError(
            "network unavailable",
            retryable=False,
            provider="test-model",
        )
        provider = build_runtime_provider(FakeChatModel(error=runtime_error))

        with self.assertRaises(ProviderGenerationError) as raised:
            provider.chat("Hello")

        self.assertIs(raised.exception.__cause__, runtime_error)
        self.assertEqual(provider.get_history(), [])
        self.assertEqual(provider.get_generation_log(), [])

    def test_rejects_blank_model(self) -> None:
        adapter = FakeChatModel(
            response=ModelResponse(message=Message.assistant("unused"))
        )
        runtime = ModelRuntime(ModelRouter({"primary": (adapter, "test-model")}))

        with self.assertRaisesRegex(ConfigurationError, "model must not be blank"):
            ModelRuntimeProvider(
                runtime=runtime,
                logical_name="primary",
                provider_name="fake",
                model=" ",
            )
