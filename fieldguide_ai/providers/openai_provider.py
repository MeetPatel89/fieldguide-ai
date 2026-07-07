from openai import OpenAI

from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        message_history: list[ChatMessage] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        super().__init__(message_history=message_history)
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[ChatMessage]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[message.to_input_item() for message in messages],
        )
        return response.output_text
