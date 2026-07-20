"""Shared terminal rendering for provider conversation state."""

import sys
from typing import TextIO

from fieldguide_ai.providers.base import LLMProvider


def write_history(
    provider: LLMProvider,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Write a provider's system prompt and conversation turns to a stream."""
    index = 1
    if provider.system_prompt is not None:
        output_stream.write(f"{index}. system: {provider.system_prompt}\n")
        index += 1
    for message in provider.get_history():
        output_stream.write(f"{index}. {message.role}: {message.content}\n")
        index += 1
