"""Integration tests for Fieldguide's model-runtime conversation dependency."""

import unittest

from model_runtime import (
    ChatSession,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ProviderUnavailableError,
    Usage,
)

from tests.session_fakes import FakeChatModel


class ModelRuntimeSessionIntegrationTest(unittest.TestCase):
    """Verify Fieldguide can consume the public runtime session contract directly."""

    def test_turn_records_runtime_response_and_provider_native_raw_value(self) -> None:
        raw = {"id": "response-123", "events": ["created"]}

        def response_factory(model_id: str, request: ModelRequest) -> ModelResponse:
            self.assertEqual(model_id, "test-model")
            return ModelResponse(
                message=Message.assistant("Generated response"),
                usage=Usage(input_tokens=10, output_tokens=4),
                raw=raw,
                model="returned-model",
            )

        adapter = FakeChatModel(response_factory=response_factory)
        runtime = ModelRuntime(ModelRouter({"openai": (adapter, "test-model")}))
        ticks = iter((0.0, 0.0125))
        provider = ChatSession(
            runtime,
            "openai",
            system_prompt="Be concise.",
            clock=lambda: next(ticks),
        )

        result = provider.complete_turn_sync("Hello")

        self.assertEqual(result.text, "Generated response")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.route, "openai")
        self.assertEqual(result.model, "returned-model")
        self.assertEqual(result.response_id, "response-123")
        self.assertEqual(result.latency_ms, 12.5)
        self.assertEqual(result.response.usage, Usage(10, 4))
        self.assertIs(result.response.raw, raw)
        self.assertIs(provider.last_record, result)
        self.assertEqual(
            adapter.requests[0][1].messages,
            (Message.system("Be concise."), Message.user("Hello")),
        )

    def test_runtime_errors_cross_fieldguide_without_app_specific_wrapping(
        self,
    ) -> None:
        runtime_error = ProviderUnavailableError(
            "network unavailable",
            retryable=False,
            provider="openai",
        )
        adapter = FakeChatModel(completion_error=runtime_error)
        provider = ChatSession(
            ModelRuntime(ModelRouter({"openai": (adapter, "test-model")})),
            "openai",
        )

        with self.assertRaises(ProviderUnavailableError) as raised:
            provider.chat_sync("Hello")

        self.assertIs(raised.exception, runtime_error)
        self.assertEqual(provider.history, ())
        self.assertEqual(provider.generation_log, ())

    def test_history_uses_model_runtime_messages(self) -> None:
        provider = ChatSession(
            ModelRuntime(
                ModelRouter(
                    {"fake": (FakeChatModel(response_text="Done"), "fake-model")}
                )
            ),
            "fake",
            history=(Message.user("Prior"), Message.assistant("Earlier answer")),
        )

        provider.chat_sync(Message.user("Next"))

        self.assertEqual(
            provider.history,
            (
                Message.user("Prior"),
                Message.assistant("Earlier answer"),
                Message.user("Next"),
                Message.assistant("Done"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
