"""Integration checks for model-runtime generation values used by Fieldguide."""

import unittest
from dataclasses import FrozenInstanceError

from model_runtime import GenerationRecord, Message, ModelResponse, Usage


class GenerationRecordIntegrationTest(unittest.TestCase):
    """Verify the runtime record exposes the telemetry Fieldguide consumes."""

    def test_exposes_response_text_usage_and_unmodified_raw_value(self) -> None:
        raw = object()
        response = ModelResponse(
            message=Message.assistant("Generated response"),
            usage=Usage(input_tokens=10, output_tokens=4),
            raw=raw,
            model="test-model",
        )
        record = GenerationRecord(
            response=response,
            route="openai",
            provider="openai",
            model="test-model",
            response_id="resp-123",
            latency_ms=12.5,
        )

        self.assertEqual(record.text, "Generated response")
        self.assertEqual(record.response.usage.total_tokens, 14)
        self.assertIs(record.response.raw, raw)

    def test_is_frozen(self) -> None:
        record = GenerationRecord(
            response=ModelResponse(message=Message.assistant("response")),
            route="openai",
            provider="openai",
            model="test-model",
            latency_ms=0,
        )

        with self.assertRaises(FrozenInstanceError):
            record.model = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
