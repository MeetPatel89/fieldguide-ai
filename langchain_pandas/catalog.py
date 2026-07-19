"""Validated catalog of named pandas dataframes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """User-supplied definition of a named dataset."""

    name: str
    dataframe: pd.DataFrame
    description: str = ""
    source_path: str | None = None


@dataclass(frozen=True)
class DatasetEntry:
    """Validated dataset entry stored in a dataframe catalog."""

    name: str
    dataframe: pd.DataFrame
    description: str
    source_path: str | None
    id_column: str | None

    @property
    def source_name(self) -> str:
        """Basename of the dataset source path."""
        if not self.source_path:
            return ""
        return Path(self.source_path).name


class DataframeCatalog:
    """Provide validated, name-based access to dataframes."""

    def __init__(self, entries: list[DatasetEntry]) -> None:
        self._entries = entries
        self._by_name = {entry.name: entry for entry in entries}

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "DataframeCatalog":
        """Validate dataset specifications and build a catalog."""
        entries: list[DatasetEntry] = []
        seen_names: set[str] = set()
        for spec in specs:
            normalized_name = spec.name.strip()
            if not normalized_name:
                raise ValueError("Dataset names must be non-empty.")
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate dataset name: {normalized_name}")
            if not isinstance(spec.dataframe, pd.DataFrame):
                raise ValueError(
                    f"Dataset '{normalized_name}' must be a pandas DataFrame."
                )

            seen_names.add(normalized_name)
            entries.append(
                DatasetEntry(
                    name=normalized_name,
                    dataframe=spec.dataframe,
                    description=spec.description.strip(),
                    source_path=spec.source_path,
                    id_column=infer_id_column(spec.dataframe),
                )
            )
        return cls(entries)

    def all(self) -> list[DatasetEntry]:
        """Return all catalog entries in insertion order."""
        return list(self._entries)

    def get(self, dataset_name: str) -> DatasetEntry:
        """Return a dataset entry by name."""
        try:
            return self._by_name[dataset_name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}."
            ) from exc

    def names(self) -> list[str]:
        """Return dataset names in insertion order."""
        return [entry.name for entry in self._entries]


def infer_id_column(dataframe: pd.DataFrame) -> str | None:
    """Infer a likely identifier column from a dataframe."""
    candidates = [
        column for column in dataframe.columns if "id" in str(column).strip().lower()
    ]
    for column in candidates:
        series = dataframe[column]
        if series.notna().all() and series.is_unique:
            return str(column)
    return str(candidates[0]) if candidates else None
