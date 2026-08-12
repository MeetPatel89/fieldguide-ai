"""Model-runtime-backed chat providers for Fieldguide AI."""

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence

from anthropic import Anthropic
from model_runtime import (
    AnthropicAdapter,
    ChatModel,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ModelRuntimeError,
    OpenAIAdapter,
)
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

from fieldguide_ai.chat import ChatMessage, GenerationResult, TokenUsage
from fieldguide_ai.errors import (
    ConfigurationError,
    ProviderDiscoveryError,
    ProviderGenerationError,
    ProviderInitializationError,
)
from fieldguide_ai.providers.base import LLMProvider, ProviderBackend

AdapterFactory = Callable[[str], ChatModel]
ModelListingFunction = Callable[[str], list[str]]
Clock = Callable[[], float]

_RAW_PAYLOAD_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _raw_payload(raw: object | None) -> dict[str, JsonValue] | None:
    """Return JSON-compatible provider metadata from a runtime response."""
    if isinstance(raw, BaseModel):
        return _RAW_PAYLOAD_ADAPTER.validate_python(raw.model_dump(mode="json"))
    if isinstance(raw, Mapping):
        return _RAW_PAYLOAD_ADAPTER.validate_python(dict(raw))
    return None


def _response_id(raw: object | None) -> str | None:
    """Extract a provider response identifier when the native payload has one."""
    value = raw.get("id") if isinstance(raw, Mapping) else getattr(raw, "id", None)
    return value if isinstance(value, str) else None


def _to_runtime_request(
    messages: Sequence[ChatMessage],
    system_prompt: str | None,
) -> ModelRequest:
    """Translate Fieldguide conversation turns into a runtime request."""
    runtime_messages: list[Message] = []
    if system_prompt is not None:
        runtime_messages.append(Message.system(system_prompt))
    runtime_messages.extend(
        Message(role=message.role, content=message.content) for message in messages
    )
    return ModelRequest(messages=runtime_messages)


def _to_generation_result(
    response: ModelResponse,
    *,
    provider_name: str,
    fallback_model: str,
    latency_ms: float,
) -> GenerationResult:
    """Translate a runtime response into Fieldguide's normalized result."""
    usage = response.usage
    return GenerationResult(
        text=response.text,
        provider=provider_name,
        model=response.model or fallback_model,
        response_id=_response_id(response.raw),
        finish_reason=response.finish_reason.value,
        usage=TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ),
        latency_ms=latency_ms,
        raw=_raw_payload(response.raw),
    )


class ModelRuntimeProvider(LLMProvider):
    """Expose model-runtime through Fieldguide's synchronous provider contract."""

    def __init__(
        self,
        runtime: ModelRuntime,
        logical_name: str,
        provider_name: str,
        model: str,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not logical_name.strip():
            raise ConfigurationError("model-runtime logical name must not be blank")
        if not provider_name.strip():
            raise ConfigurationError("provider name must not be blank")
        if not model.strip():
            raise ConfigurationError(f"{provider_name} model must not be blank")
        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self._runtime = runtime
        self._logical_name = logical_name
        self._provider_name = provider_name
        self._model = model
        self._clock = clock or time.perf_counter

    @property
    def model(self) -> str:
        """The configured provider model ID."""
        return self._model

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        """Generate a response through the asynchronous shared runtime."""
        request = _to_runtime_request(messages, self.system_prompt)
        try:
            started = self._clock()
            response = asyncio.run(self._runtime.complete(self._logical_name, request))
            latency_ms = (self._clock() - started) * 1000
        except ModelRuntimeError as error:
            raise ProviderGenerationError(
                f"{self._provider_name} generation failed for model {self._model!r}"
            ) from error

        return self._record_generation(
            _to_generation_result(
                response,
                provider_name=self._provider_name,
                fallback_model=self._model,
                latency_ms=latency_ms,
            )
        )


class ModelRuntimeBackend(ProviderBackend):
    """Construct runtime chat adapters while retaining SDK model discovery."""

    def __init__(
        self,
        provider_name: str,
        api_key: str | None,
        adapter_factory: AdapterFactory,
        model_listing: ModelListingFunction,
        *,
        credential_name: str | None = None,
    ) -> None:
        if not provider_name.strip():
            raise ConfigurationError("provider name must not be blank")
        self._provider_name = provider_name
        self._api_key = api_key
        self._adapter_factory = adapter_factory
        self._model_listing = model_listing
        self._credential_name = credential_name or f"{provider_name.upper()}_API_KEY"

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise ConfigurationError(
                f"{self._credential_name} is not set. Add it to your .env file."
            )
        return self._api_key

    def build_provider(
        self,
        model: str,
        *,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> LLMProvider:
        """Build a synchronous provider around a single model-runtime route."""
        if not model.strip():
            raise ConfigurationError(f"{self._provider_name} model must not be blank")
        try:
            adapter = self._adapter_factory(self._require_api_key())
            router = ModelRouter(
                {self._provider_name: (adapter, model)},
            )
            runtime = ModelRuntime(router)
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderInitializationError(
                f"{self._provider_name} client initialization failed"
            ) from error
        return ModelRuntimeProvider(
            runtime=runtime,
            logical_name=self._provider_name,
            provider_name=self._provider_name,
            model=model,
            message_history=message_history,
            system_prompt=system_prompt,
        )

    def list_models(self) -> list[str]:
        """List models with the provider SDK's discovery API."""
        try:
            return self._model_listing(self._require_api_key())
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderDiscoveryError(
                f"{self._provider_name} model discovery failed"
            ) from error


def _create_openai_adapter(api_key: str) -> ChatModel:
    return OpenAIAdapter(api_key=api_key)


def _create_anthropic_adapter(api_key: str) -> ChatModel:
    return AnthropicAdapter(api_key=api_key)


def _list_openai_models(api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key)
    return sorted(model.id for model in client.models.list().data)


def _list_anthropic_models(api_key: str) -> list[str]:
    client = Anthropic(api_key=api_key)
    return sorted(model.id for model in client.models.list().data)


def openai_backend(api_key: str | None) -> ModelRuntimeBackend:
    """Build the runtime backend configuration for OpenAI."""
    return ModelRuntimeBackend(
        provider_name="openai",
        api_key=api_key,
        adapter_factory=_create_openai_adapter,
        model_listing=_list_openai_models,
        credential_name="OPENAI_API_KEY",
    )


def anthropic_backend(api_key: str | None) -> ModelRuntimeBackend:
    """Build the runtime backend configuration for Anthropic."""
    return ModelRuntimeBackend(
        provider_name="anthropic",
        api_key=api_key,
        adapter_factory=_create_anthropic_adapter,
        model_listing=_list_anthropic_models,
        credential_name="ANTHROPIC_API_KEY",
    )
