"""Validated catalog of named pandas dataframes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd


class DatasetCatalogError(ValueError):
    """Raised when dataframe catalog configuration or lookup is invalid."""


@dataclass(frozen=True, init=False, eq=False)
class DatasetSpec:
    """Validated user-supplied definition of a named dataset."""

    name: str
    _dataframe: pd.DataFrame = field(repr=False, compare=False)
    description: str = ""
    source_path: str | None = None

    def __init__(
        self,
        name: str,
        dataframe: pd.DataFrame,
        description: str = "",
        source_path: str | None = None,
    ) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise DatasetCatalogError("Dataset names must be non-empty.")
        if not isinstance(dataframe, pd.DataFrame):
            raise DatasetCatalogError(
                f"Dataset '{normalized_name}' must be a pandas DataFrame."
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "_dataframe", dataframe.copy(deep=True))
        object.__setattr__(self, "description", description.strip())
        object.__setattr__(self, "source_path", source_path)

    @property
    def dataframe(self) -> pd.DataFrame:
        """An independent copy of the supplied dataframe."""
        return self._dataframe.copy(deep=True)


@dataclass(frozen=True, init=False, eq=False)
class DatasetEntry:
    """Encapsulated dataset entry stored in a dataframe catalog."""

    name: str
    _dataframe: pd.DataFrame = field(repr=False, compare=False)
    description: str
    source_path: str | None
    id_column: str | None

    def __init__(
        self,
        name: str,
        dataframe: pd.DataFrame,
        description: str,
        source_path: str | None,
        id_column: str | None,
    ) -> None:
        if not name.strip():
            raise DatasetCatalogError("Dataset entry names must be non-empty.")
        if id_column is not None and id_column not in dataframe.columns:
            raise DatasetCatalogError(
                f"ID column {id_column!r} is not present in {name!r}."
            )
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "_dataframe", dataframe.copy(deep=True))
        object.__setattr__(self, "description", description.strip())
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "id_column", id_column)

    @property
    def dataframe(self) -> pd.DataFrame:
        """An independent copy of the cataloged dataframe."""
        return self._dataframe.copy(deep=True)

    @property
    def source_name(self) -> str:
        """Basename of the dataset source path."""
        if not self.source_path:
            return ""
        return Path(self.source_path).name


class DataframeCatalog:
    """Provide validated, name-based access to dataframes."""

    def __init__(self, entries: list[DatasetEntry]) -> None:
        by_name = {entry.name: entry for entry in entries}
        if len(by_name) != len(entries):
            raise DatasetCatalogError("Dataset entry names must be unique.")
        self._entries = tuple(entries)
        self._by_name: Mapping[str, DatasetEntry] = MappingProxyType(by_name)

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "DataframeCatalog":
        """Validate dataset specifications and build a catalog."""
        entries: list[DatasetEntry] = []
        seen_names: set[str] = set()
        for spec in specs:
            if spec.name in seen_names:
                raise DatasetCatalogError(f"Duplicate dataset name: {spec.name}")

            dataframe = spec.dataframe
            seen_names.add(spec.name)
            entries.append(
                DatasetEntry(
                    name=spec.name,
                    dataframe=dataframe,
                    description=spec.description,
                    source_path=spec.source_path,
                    id_column=infer_id_column(dataframe),
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
            raise DatasetCatalogError(
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
