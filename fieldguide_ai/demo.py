"""Demo chat history for Fieldguide AI."""

from fieldguide_ai.messages import ChatMessage

DEFAULT_SYSTEM_PROMPT = (
    "You are a super expert in linear algebra that likes answering questions in detail."
)


def build_demo_messages() -> list[ChatMessage]:
    """Build a demo chat history."""
    return [
        ChatMessage(
            role="user",
            content=(
                "I am planning to become a linear algebra expert. "
                "What are the best books to read? "
                "Assume I have basic understanding of linear algebra. "
                "I am thinking about starting with Meyer's"
            ),
        ),
    ]


def build_system_prompt() -> str:
    """Build a system prompt."""
    return DEFAULT_SYSTEM_PROMPT
