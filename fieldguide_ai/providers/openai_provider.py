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
            input=[message.to_provider_message("openai") for message in messages],
        )
        return response.output_text


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    response_text = provider.generate(
        [
            ChatMessage(
                role="user",
                content="Write a motivational message for learners studying AI development.",
            )
        ]
    )
    print(response_text)


if __name__ == "__main__":
    main()
