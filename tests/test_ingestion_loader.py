import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fieldguide_ai.errors import DocumentLoadError
from fieldguide_ai.ingestion.loader import (
    load_markdown_document,
    load_markdown_documents,
)


class MarkdownLoaderTest(unittest.TestCase):
    def test_translates_file_read_errors(self) -> None:
        missing_path = Path("does-not-exist.md")

        with self.assertRaises(DocumentLoadError) as raised:
            load_markdown_document(missing_path)

        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_loads_frontmatter_and_body(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "doc.md"
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "doc_id: DOC-1",
                        "title: Test Document",
                        "service_id: null",
                        "related_records:",
                        "  - KI-1",
                        "  - INC-2",
                        "---",
                        "",
                        "# Test Document",
                        "",
                        "Body text.",
                    ]
                ),
                encoding="utf-8",
            )

            document = load_markdown_document(path)

        self.assertEqual(document.doc_id, "DOC-1")
        self.assertEqual(document.title, "Test Document")
        self.assertIsNone(document.metadata["service_id"])
        self.assertEqual(document.metadata["related_records"], ["KI-1", "INC-2"])
        self.assertEqual(document.body, "# Test Document\n\nBody text.")

    def test_loads_documents_in_stable_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "b.md").write_text("# B", encoding="utf-8")
            (root / "a.md").write_text("# A", encoding="utf-8")

            documents = load_markdown_documents(root)

        self.assertEqual(
            [document.source_path.name for document in documents], ["a.md", "b.md"]
        )

    def test_loads_nested_frontmatter(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incident.md"
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "doc_id: INC-1",
                        "visibility: internal",
                        "servicenow:",
                        "  table: incident",
                        "  number: INC000001",
                        "  priority: 2",
                        "  sla_breached: false",
                        "  task_numbers:",
                        "    - INCT0001001",
                        "    - INCT0001002",
                        "  visibility: customer_safe",
                        "---",
                        "",
                        "# Incident",
                    ]
                ),
                encoding="utf-8",
            )

            document = load_markdown_document(path)

        self.assertEqual(document.metadata["visibility"], "internal")
        self.assertEqual(
            document.metadata["servicenow"],
            {
                "table": "incident",
                "number": "INC000001",
                "priority": 2,
                "sla_breached": False,
                "task_numbers": ["INCT0001001", "INCT0001002"],
                "visibility": "customer_safe",
            },
        )
        self.assertNotIn("table", document.metadata)
        self.assertNotIn("number", document.metadata)
