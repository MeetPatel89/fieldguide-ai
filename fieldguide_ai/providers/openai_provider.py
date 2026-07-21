"""OpenAI provider for the Fieldguide AI."""

import os
import time
from collections.abc import Callable, Sequence

from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
)

from fieldguide_ai.chat import ChatMessage, GenerationResult, TokenUsage
from fieldguide_ai.errors import (
    ConfigurationError,
    ProviderDiscoveryError,
    ProviderGenerationError,
    ProviderInitializationError,
)
from fieldguide_ai.providers.base import LLMProvider, ProviderBackend


def to_openai_input(
    messages: Sequence[ChatMessage],
    system_prompt: str | None = None,
) -> ResponseInputParam:
    """Convert a system prompt and conversation turns to OpenAI input items."""
    items: ResponseInputParam = []
    if system_prompt:
        items.append(EasyInputMessageParam(role="system", content=system_prompt))
    items.extend(
        EasyInputMessageParam(role=message.role, content=message.content)
        for message in messages
    )
    return items


def from_openai_response(
    response: Response,
    *,
    model: str,
    latency_ms: float | None = None,
) -> GenerationResult:
    """Normalize an OpenAI Responses API payload into ``GenerationResult``."""
    usage = None
    usage_payload = getattr(response, "usage", None)
    if usage_payload is not None:
        usage = TokenUsage(
            input_tokens=getattr(usage_payload, "input_tokens", None),
            output_tokens=getattr(usage_payload, "output_tokens", None),
            total_tokens=getattr(usage_payload, "total_tokens", None),
        )

    raw = response.model_dump(mode="json")
    return GenerationResult(
        text=getattr(response, "output_text", None) or "",
        provider="openai",
        model=model,
        response_id=getattr(response, "id", None),
        finish_reason=getattr(response, "status", None),
        usage=usage,
        latency_ms=latency_ms,
        raw=raw,
    )


class OpenAIProvider(LLMProvider):
    """OpenAI provider for the Fieldguide AI."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("OpenAI model must not be blank")
        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self._model = model
        self._client = client
        self._clock = clock or time.perf_counter

    @property
    def model(self) -> str:
        """The configured OpenAI model ID."""
        return self._model

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        """Generate a response from the model."""
        try:
            started = self._clock()
            response = self._client.responses.create(
                model=self._model,
                input=to_openai_input(messages, system_prompt=self.system_prompt),
            )
            latency_ms = (self._clock() - started) * 1000
            result = from_openai_response(
                response,
                model=self._model,
                latency_ms=latency_ms,
            )
        except Exception as error:
            raise ProviderGenerationError(
                f"OpenAI generation failed for model {self._model!r}"
            ) from error
        return self._record_generation(result)


class OpenAIBackend(ProviderBackend):
    """Own OpenAI SDK-client creation, discovery, and chat construction."""

    def __init__(
        self,
        api_key: str | None,
        client_factory: Callable[..., OpenAI] | None = None,
    ) -> None:
        self._api_key = api_key
        self._client_factory = client_factory or OpenAI

    def _build_client(self) -> OpenAI:
        if not self._api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return self._client_factory(api_key=self._api_key)

    def build_provider(
        self,
        model: str,
        *,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> LLMProvider:
        """Build an OpenAI chat provider for a model."""
        try:
            client = self._build_client()
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderInitializationError(
                "OpenAI client initialization failed"
            ) from error
        return OpenAIProvider(
            client=client,
            model=model,
            message_history=message_history,
            system_prompt=system_prompt,
        )

    def list_models(self) -> list[str]:
        """List models visible to the configured OpenAI account."""
        try:
            client = self._build_client()
            return sorted(model.id for model in client.models.list().data)
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderDiscoveryError("OpenAI model discovery failed") from error


def main() -> None:
    """Run the main function."""
    from dotenv import load_dotenv

    load_dotenv()
    backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
    provider = backend.build_provider("gpt-5-nano")
    provider.set_system_prompt(
        "You are a helpful assistant that likes keeping things short and concise."
    )
    result = provider.generate(
        [
            ChatMessage(
                role="user",
                content=("Tell me about yourself."),
            )
        ]
    )
    print(result.text)
    print(backend.list_models())


if __name__ == "__main__":
    main()
