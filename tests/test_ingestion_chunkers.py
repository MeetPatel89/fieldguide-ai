from pathlib import Path
import unittest

from fieldguide_ai.ingestion.chunkers import MarkdownSectionChunker, count_words, split_markdown_sections
from fieldguide_ai.ingestion.models import MarkdownDocument


class MarkdownSectionChunkerTest(unittest.TestCase):
    def test_splits_markdown_on_h2_sections(self) -> None:
        sections = split_markdown_sections(
            "\n".join(
                [
                    "# API Gateway Runbook",
                    "",
                    "Intro text.",
                    "",
                    "## Initial triage",
                    "",
                    "Collect the client ID.",
                    "",
                    "## Escalation criteria",
                    "",
                    "Escalate to Platform Engineering.",
                ]
            ),
            fallback_title="Fallback",
        )

        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0].section_path, ("API Gateway Runbook",))
        self.assertIn("Intro text.", sections[0].content)
        self.assertEqual(sections[1].section_path, ("API Gateway Runbook", "Initial triage"))
        self.assertIn("Collect the client ID.", sections[1].content)
        self.assertEqual(sections[2].section_path, ("API Gateway Runbook", "Escalation criteria"))

    def test_chunk_document_attaches_metadata_to_each_chunk(self) -> None:
        document = MarkdownDocument(
            source_path=Path("data/raw/runbook.md"),
            metadata={
                "doc_id": "RUNBOOK-1",
                "title": "API Gateway Runbook",
                "doc_type": "runbook",
                "service_id": "svc_api",
                "owner_group": "Platform Engineering",
                "related_records": ["KI-1"],
                "servicenow": {"table": "kb_knowledge", "number": "KB0001"},
            },
            body="\n".join(
                [
                    "# API Gateway Runbook",
                    "",
                    "## Initial triage",
                    "",
                    "Collect client ID and timestamp.",
                ]
            ),
        )

        chunks = MarkdownSectionChunker().chunk_document(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "RUNBOOK-1::chunk-0000")
        self.assertEqual(chunks[0].doc_id, "RUNBOOK-1")
        self.assertEqual(chunks[0].section_path, ("API Gateway Runbook", "Initial triage"))
        self.assertEqual(chunks[0].metadata["doc_type"], "runbook")
        self.assertEqual(chunks[0].metadata["service_id"], "svc_api")
        self.assertEqual(chunks[0].metadata["owner_group"], "Platform Engineering")
        self.assertEqual(chunks[0].metadata["related_records"], ["KI-1"])
        self.assertEqual(chunks[0].metadata["servicenow"]["number"], "KB0001")
        self.assertEqual(chunks[0].metadata["section_title"], "Initial triage")

    def test_splits_large_sections_with_overlap(self) -> None:
        long_paragraph = " ".join(f"word{index}" for index in range(230))
        document = MarkdownDocument(
            source_path=Path("large.md"),
            metadata={"doc_id": "DOC-LARGE", "title": "Large Document"},
            body=f"# Large Document\n\n## Long section\n\n{long_paragraph}",
        )

        chunks = MarkdownSectionChunker(max_words=100, overlap_words=10).chunk_document(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(count_words(chunk.content) <= 100 for chunk in chunks))
        self.assertEqual(chunks[0].chunk_id, "DOC-LARGE::chunk-0000")
        self.assertEqual(chunks[1].chunk_id, "DOC-LARGE::chunk-0001")

    def test_chunk_record_is_serializable(self) -> None:
        document = MarkdownDocument(
            source_path=Path("known_issue.md"),
            metadata={"doc_id": "KI-1", "title": "Known Issue", "doc_type": "known_issue"},
            body="# Known Issue\n\n## Summary\n\nUnexpected 429 errors.",
        )

        record = MarkdownSectionChunker().chunk_document(document)[0].to_record()

        self.assertEqual(record["chunk_id"], "KI-1::chunk-0000")
        self.assertEqual(record["source_path"], "known_issue.md")
        self.assertEqual(record["section_path"], ["Known Issue", "Summary"])
