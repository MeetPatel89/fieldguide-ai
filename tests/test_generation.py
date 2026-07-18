import unittest

from pydantic import ValidationError

from fieldguide_ai.generation import GenerationResult, TokenUsage


class TokenUsageValidationTest(unittest.TestCase):
    def test_rejects_negative_counts(self) -> None:
        with self.assertRaises(ValidationError):
            TokenUsage(input_tokens=-1)

    def test_rejects_type_coercion(self) -> None:
        with self.assertRaises(ValidationError):
            TokenUsage.model_validate({"input_tokens": "10"})


class GenerationResultValidationTest(unittest.TestCase):
    def test_round_trips_as_json(self) -> None:
        result = GenerationResult(
            text="Generated response",
            provider="openai",
            model="test-model",
            response_id="resp_123",
            finish_reason="completed",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
            ),
            latency_ms=12.5,
            raw={"id": "resp_123", "events": [1, True, None]},
        )

        restored = GenerationResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored, result)

    def test_validates_nested_usage_from_serialized_data(self) -> None:
        result = GenerationResult.model_validate(
            {
                "text": "Generated response",
                "provider": "openai",
                "model": "test-model",
                "usage": {"input_tokens": 10},
            }
        )

        self.assertEqual(result.usage, TokenUsage(input_tokens=10))

    def test_rejects_blank_provider_and_model(self) -> None:
        for field in ("provider", "model"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                payload = {
                    "text": "Generated response",
                    "provider": "openai",
                    "model": "test-model",
                }
                payload[field] = "   "
                GenerationResult.model_validate(payload)

    def test_rejects_negative_or_non_finite_latency(self) -> None:
        for latency_ms in (-0.1, float("inf"), float("nan")):
            with (
                self.subTest(latency_ms=latency_ms),
                self.assertRaises(ValidationError),
            ):
                GenerationResult(
                    text="Generated response",
                    provider="openai",
                    model="test-model",
                    latency_ms=latency_ms,
                )

    def test_rejects_non_json_raw_values(self) -> None:
        with self.assertRaises(ValidationError):
            GenerationResult.model_validate(
                {
                    "text": "Generated response",
                    "provider": "openai",
                    "model": "test-model",
                    "raw": {"invalid": object()},
                }
            )

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GenerationResult.model_validate(
                {
                    "text": "Generated response",
                    "provider": "openai",
                    "model": "test-model",
                    "unexpected": True,
                }
            )

    def test_is_immutable(self) -> None:
        result = GenerationResult(
            text="Generated response",
            provider="openai",
            model="test-model",
        )

        with self.assertRaises(ValidationError):
            result.text = "changed"  # type: ignore[misc]
