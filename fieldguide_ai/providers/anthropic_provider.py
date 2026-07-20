"""Anthropic provider for the Fieldguide AI."""

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from fieldguide_ai.chat import ChatMessage, GenerationResult, TokenUsage
from fieldguide_ai.errors import (
    ConfigurationError,
    ProviderDiscoveryError,
    ProviderGenerationError,
    ProviderInitializationError,
)
from fieldguide_ai.providers.base import LLMProvider, ProviderBackend


def to_anthropic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert conversation turns to Anthropic Messages API message dicts."""
    return [{"role": message.role, "content": message.content} for message in messages]


def from_anthropic_response(
    response: Message,
    *,
    model: str,
    latency_ms: float | None = None,
) -> GenerationResult:
    """Normalize an Anthropic Messages API payload into ``GenerationResult``."""
    content = getattr(response, "content", None) or []
    text = "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )

    usage = None
    usage_payload = getattr(response, "usage", None)
    if usage_payload is not None:
        input_tokens = getattr(usage_payload, "input_tokens", None)
        output_tokens = getattr(usage_payload, "output_tokens", None)
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    raw = response.model_dump(mode="json")

    return GenerationResult(
        text=text,
        provider="anthropic",
        model=model,
        response_id=getattr(response, "id", None),
        finish_reason=getattr(response, "stop_reason", None),
        usage=usage,
        latency_ms=latency_ms,
        raw=raw,
    )


class AnthropicProvider(LLMProvider):
    """Anthropic provider for the Fieldguide AI."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("Anthropic model must not be blank")
        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self._model = model
        self._client = client
        self._clock = clock or time.perf_counter

    @property
    def model(self) -> str:
        """The configured Anthropic model ID."""
        return self._model

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        """Generate a response from the model."""
        create_kwargs: dict[str, Any] = {
            "max_tokens": 1000,
            "model": self._model,
            "messages": to_anthropic_messages(messages),
        }
        if self.system_prompt is not None:
            create_kwargs["system"] = self.system_prompt

        try:
            started = self._clock()
            response = self._client.messages.create(**create_kwargs)
            latency_ms = (self._clock() - started) * 1000
            result = from_anthropic_response(
                response,
                model=self._model,
                latency_ms=latency_ms,
            )
        except Exception as error:
            raise ProviderGenerationError(
                f"Anthropic generation failed for model {self._model!r}"
            ) from error
        return self._record_generation(result)


class AnthropicBackend(ProviderBackend):
    """Own Anthropic SDK-client creation, discovery, and chat construction."""

    def __init__(
        self,
        api_key: str | None,
        client_factory: Callable[..., Anthropic] | None = None,
    ) -> None:
        self._api_key = api_key
        self._client_factory = client_factory or Anthropic

    def _build_client(self) -> Anthropic:
        if not self._api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        return self._client_factory(api_key=self._api_key)

    def build_provider(
        self,
        model: str,
        *,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> LLMProvider:
        """Build an Anthropic chat provider for a model."""
        try:
            client = self._build_client()
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderInitializationError(
                "Anthropic client initialization failed"
            ) from error
        return AnthropicProvider(
            client=client,
            model=model,
            message_history=message_history,
            system_prompt=system_prompt,
        )

    def list_models(self) -> list[str]:
        """List models visible to the configured Anthropic account."""
        try:
            client = self._build_client()
            return sorted(model.id for model in client.models.list().data)
        except ConfigurationError:
            raise
        except Exception as error:
            raise ProviderDiscoveryError("Anthropic model discovery failed") from error


def main() -> None:
    """Run the main function."""
    from dotenv import load_dotenv

    load_dotenv()
    backend = AnthropicBackend(api_key=os.getenv("ANTHROPIC_API_KEY"))
    provider = backend.build_provider("claude-haiku-4-5-20251001")
    provider.set_system_prompt("Answer briefly.")
    result = provider.generate(
        [
            ChatMessage(
                role="user",
                content="Hello, how are you? Please tell me about yourself.",
            )
        ]
    )
    print(result.text)
    print(backend.list_models())


if __name__ == "__main__":
    main()
