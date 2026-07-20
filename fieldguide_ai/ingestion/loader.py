"""
Loader for markdown documents.

This module provides functions to load markdown documents and parse their frontmatter.
"""

import re
from pathlib import Path
from typing import Any, Iterable

from fieldguide_ai.errors import DocumentLoadError
from fieldguide_ai.ingestion.models import MarkdownDocument

FRONTMATTER_BOUNDARY = "---"


def load_markdown_document(path: str | Path) -> MarkdownDocument:
    """
    Load a markdown document from a file.

    Parameters
    ----------
    path : str or Path
        The path to the markdown document.

    Returns
    -------
    document : MarkdownDocument
        The loaded markdown document.
    """
    source_path = Path(path)
    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DocumentLoadError(
            f"could not read Markdown document at {source_path}"
        ) from error
    metadata, body = parse_markdown_frontmatter(raw_text)
    return MarkdownDocument(
        source_path=source_path, body=body.strip(), metadata=metadata
    )


def load_markdown_documents(root: str | Path) -> list[MarkdownDocument]:
    """
    Load all markdown documents from a directory.

    Parameters
    ----------
    root : str or Path
        The path to the root directory.

    Returns
    -------
    documents : list[MarkdownDocument]
        The loaded markdown documents.
    """
    root_path = Path(root)
    try:
        paths = sorted(root_path.rglob("*.md"))
    except OSError as error:
        raise DocumentLoadError(
            f"could not discover Markdown documents under {root_path}"
        ) from error
    return [load_markdown_document(path) for path in paths]


def parse_markdown_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """
    Parse the frontmatter of a markdown document.

    Parameters
    ----------
    raw_text : str
        The raw text of the markdown document.

    Returns
    -------
    metadata : dict[str, Any]
        The metadata of the markdown document.
    body : str
        The body of the markdown document.
    """
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
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, metadata)]
    pending_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        while stack and indent <= stack[-1][0]:
            stack.pop()

        if pending_key is not None and indent > pending_key[0]:
            pending_indent, pending_parent, pending_name = pending_key
            container: dict[str, Any] | list[Any]
            if stripped.startswith("- "):
                container = []
            else:
                container = {}
            pending_parent[pending_name] = container
            stack.append((pending_indent, container))
            pending_key = None

        parent = stack[-1][1]

        if stripped.startswith("- "):
            if isinstance(parent, list):
                parent.append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in line:
            pending_key = None
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if not isinstance(parent, dict):
            pending_key = None
            continue

        if value == "":
            parent[key] = {}
            pending_key = (indent, parent, key)
            continue

        parent[key] = _parse_scalar(value)
        pending_key = None

    return metadata


def _parse_scalar(value: str) -> object:
    normalized_value = value.lower()
    if normalized_value == "null":
        return None
    if normalized_value in {"true", "false"}:
        return normalized_value == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if re.match(r"^-?\d+$", value):
        return int(value)
    if re.match(r"^-?\d+\.\d+$", value):
        return float(value)
    return value


if __name__ == "__main__":
    file_path = f"{Path(__file__).parent.parent.parent}/data/corpora/nautilus/raw/"
    documents = load_markdown_documents(file_path)
    for document in documents:
        print("--------------------------------")
        print(f"Document ID: {document.doc_id}")
        print(f"Document Title: {document.title}")
        print(document)
        print("--------------------------------")
