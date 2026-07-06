import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

@dataclass
class RandomClass:
    some_field: str
    some_other_field: int


@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation."""
    role: str
    content: str

    def to_input_item(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[ChatMessage]) -> str:
        """Return model output for a chat-style prompt."""


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[ChatMessage]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[message.to_input_item() for message in messages],
        )
        return response.output_text


def build_demo_messages() -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a super expert in linear algebra that likes answering questions in detail."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "I am planning to become a linear algebra expert. What are the best books to read?Assume I have basic understanding of linear algebra."
            ),
        ),
    ]


def main() -> None:
    load_dotenv()
    print("Hello from some-project!")

    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-5-nano",
    )
    response_text = provider.generate(build_demo_messages())
    print(response_text)


if __name__ == "__main__":
    random_obj = RandomClass(some_field="some value", some_other_field=123)
    print(random_obj)
    print(help(RandomClass))
    print(help(ChatMessage))
