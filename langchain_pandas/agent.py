"""Factory for the dataframe-backed LangChain agent."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from langchain_pandas.catalog import DataframeCatalog
from langchain_pandas.tools import build_tools


def create_pandas_like_agent(
    model: object,
    catalog: DataframeCatalog,
    system_prompt: SystemMessage | None = None,
    debug: bool = False,
) -> object:
    """Create an agent equipped with tools for a dataframe catalog."""
    prompt = system_prompt or _default_system_prompt(catalog)
    return create_agent(
        model=model,
        tools=build_tools(catalog),
        system_prompt=prompt,
        debug=debug,
        name="pandas_like_agent",
    )


def _default_system_prompt(catalog: DataframeCatalog) -> SystemMessage:
    dataset_lines = []
    for entry in catalog.all():
        description = entry.description or "No description provided."
        dataset_lines.append(f"- {entry.name}: {description}")
    dataset_summary = "\n".join(dataset_lines)
    return SystemMessage(
        content=(
            "You are a data assistant that answers questions only from the available "
            "dataframes.\n"
            "Use tools before answering. Do not answer from general knowledge.\n"
            "Choose the most relevant dataset, mention the dataset name in the final "
            "answer, and cite row-level evidence when available.\n"
            "Available datasets:\n"
            f"{dataset_summary}"
        )
    )
