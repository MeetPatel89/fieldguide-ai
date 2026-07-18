"""OpenAI provider for the Fieldguide AI."""

import time

from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
)

from fieldguide_ai.generation import GenerationResult, TokenUsage
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


def to_openai_input(
    messages: list[ChatMessage],
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
        api_key: str | None,
        model: str,
        message_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Generate a response from the model."""
        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            input=to_openai_input(messages, system_prompt=self.system_prompt),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self._record_generation(
            from_openai_response(
                response,
                model=self.model,
                latency_ms=latency_ms,
            )
        )

    def list_models(self) -> list[str]:
        """List available models."""
        return sorted(model.id for model in self.client.models.list().data)


def main() -> None:
    """Run the main function."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        system_prompt=(
            "You are a helpful assistant that likes keeping things short and concise."
        ),
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
    print(provider.list_models())


if __name__ == "__main__":
    main()
