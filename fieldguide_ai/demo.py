from fieldguide_ai.messages import ChatMessage


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
                "I am planning to become a linear algebra expert. What are the best books to read? "
                "Assume I have basic understanding of linear algebra. I am thinking about starting with Meyer's"
            ),
        ),
    ]


def build_system_message() -> ChatMessage:
    return ChatMessage(
        role="system",
        content=(
            "You are a super expert in linear algebra that likes answering questions in detail."
        ),
    )
