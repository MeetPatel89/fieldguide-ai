import io
import unittest

from fieldguide_ai.cli import print_history, run_chat_loop
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def generate(self, messages: list[ChatMessage]) -> str:
        user_messages = [message.content for message in messages if message.role == "user"]
        return f"reply to {user_messages[-1]}"


class CliTest(unittest.TestCase):
    def test_chat_loop_maintains_history_across_turns(self) -> None:
        provider = FakeProvider()
        input_stream = io.StringIO("First question\nFollow up\n:quit\n")
        output_stream = io.StringIO()

        run_chat_loop(provider, input_stream=input_stream, output_stream=output_stream)

        self.assertEqual(
            [message.role for message in provider.get_history()],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertIn("Assistant> reply to First question", output_stream.getvalue())
        self.assertIn("Assistant> reply to Follow up", output_stream.getvalue())

    def test_chat_loop_can_clear_history_but_keeps_system_prompt(self) -> None:
        provider = FakeProvider()
        input_stream = io.StringIO("Question\n:clear\n:quit\n")
        output_stream = io.StringIO()

        run_chat_loop(provider, input_stream=input_stream, output_stream=output_stream)

        self.assertEqual(provider.get_history(), [provider.get_history()[0]])
        self.assertEqual(provider.get_history()[0].role, "system")
        self.assertIn("History cleared.", output_stream.getvalue())

    def test_print_history_writes_numbered_messages(self) -> None:
        provider = FakeProvider(
            message_history=[
                ChatMessage(role="system", content="Rules"),
                ChatMessage(role="user", content="Hello"),
            ]
        )
        output_stream = io.StringIO()

        print_history(provider, output_stream=output_stream)

        self.assertEqual(output_stream.getvalue(), "1. system: Rules\n2. user: Hello\n")
