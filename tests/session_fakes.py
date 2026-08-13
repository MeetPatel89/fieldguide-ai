"""Reusable model-runtime fakes for Fieldguide integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

from model_runtime import (
    ChatSession,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ModelRuntimeError,
    StreamEvent,
)

ResponseFactory = Callable[[str, ModelRequest], ModelResponse]


class FakeChatModel:
    """Network-free chat and discovery adapter with recorded requests."""

    capabilities = ModelCapabilities(streaming=False)

    def __init__(
        self,
        response_text: str = "response",
        *,
        discovered_models: Sequence[str] = ("fake-model",),
        completion_error: ModelRuntimeError | None = None,
        discovery_error: ModelRuntimeError | None = None,
        response_factory: ResponseFactory | None = None,
    ) -> None:
        self.response_text = response_text
        self.discovered_models = list(discovered_models)
        self.completion_error = completion_error
        self.discovery_error = discovery_error
        self.response_factory = response_factory
        self.requests: list[tuple[str, ModelRequest]] = []
        self.discovery_calls = 0

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Record a request and return or raise configured behavior."""
        self.requests.append((model_id, request))
        if self.completion_error is not None:
            raise self.completion_error
        if self.response_factory is not None:
            return self.response_factory(model_id, request)
        return ModelResponse(
            message=Message.assistant(self.response_text),
            model=model_id,
        )

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Reject streaming, which Fieldguide's sessions do not use."""
        raise NotImplementedError
        yield

    async def list_models(self) -> list[str]:
        """Return or raise configured discovery behavior."""
        self.discovery_calls += 1
        if self.discovery_error is not None:
            raise self.discovery_error
        return list(self.discovered_models)


class RecordingAdapterFactory:
    """Return one fake adapter while recording supplied credentials."""

    def __init__(self, adapter: FakeChatModel) -> None:
        self.adapter = adapter
        self.api_keys: list[str] = []

    def __call__(self, api_key: str) -> FakeChatModel:
        """Record the key and return the configured adapter."""
        self.api_keys.append(api_key)
        return self.adapter


def make_session(
    adapter: FakeChatModel | None = None,
    *,
    route: str = "fake",
    model: str = "fake-model",
    history: Sequence[Message] = (),
    system_prompt: str | None = None,
) -> ChatSession:
    """Build a single-route session around a fake adapter."""
    selected_adapter = adapter or FakeChatModel()
    runtime = ModelRuntime(ModelRouter({route: (selected_adapter, model)}))
    return ChatSession(
        runtime,
        route,
        history=history,
        system_prompt=system_prompt,
    )


def selected_model(session: ChatSession) -> str:
    """Return the static provider model configured for a test session."""
    return session.runtime.router.resolve(session.route).model_id
