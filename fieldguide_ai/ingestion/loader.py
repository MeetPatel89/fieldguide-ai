from pathlib import Path
from typing import Any, Iterable

from fieldguide_ai.ingestion.models import MarkdownDocument

FRONTMATTER_BOUNDARY = "---"


def load_markdown_document(path: str | Path) -> MarkdownDocument:
    source_path = Path(path)
    raw_text = source_path.read_text(encoding="utf-8")
    metadata, body = parse_markdown_frontmatter(raw_text)
    return MarkdownDocument(source_path=source_path, body=body.strip(), metadata=metadata)


def load_markdown_documents(root: str | Path) -> list[MarkdownDocument]:
    root_path = Path(root)
    return [load_markdown_document(path) for path in sorted(root_path.rglob("*.md"))]


def parse_markdown_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        return {}, raw_text

    closing_index = _find_closing_frontmatter(lines)
    if closing_index is None:
        return {}, raw_text

    frontmatter = lines[1:closing_index]
    body = "\n".join(lines[closing_index + 1 :])
    return _parse_simple_yaml(frontmatter), body


def _find_closing_frontmatter(lines: list[str]) -> int | None:
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_BOUNDARY:
            return index
    return None


def _parse_simple_yaml(lines: Iterable[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    active_list_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and active_list_key is not None:
            metadata[active_list_key].append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in line:
            active_list_key = None
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value == "":
            metadata[key] = []
            active_list_key = key
            continue

        metadata[key] = _parse_scalar(value)
        active_list_key = None

    return metadata


def _parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value in {"true", "false"}:
        return value == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
