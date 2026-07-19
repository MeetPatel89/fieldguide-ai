"""Interactive entry point for the dataframe-backed LangChain agent."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_pandas import DataframeCatalog, DatasetSpec, create_pandas_like_agent

load_dotenv()

COMMON_PATH = "data/corpora/nautilus/misc"
DATASET_METADATA = {
    "saas_docs.csv": (
        "SaaS Docs",
        "Product and API documentation for SaaS platform features and limits.",
    ),
    "credit_card_terms.csv": (
        "Credit Card Terms",
        "Credit card terms, APR details, fees, and account policies.",
    ),
    "hospital_policy.csv": (
        "Hospital Policy",
        "Hospital operations, compliance rules, and patient care policies.",
    ),
    "ecommerce_faqs.csv": (
        "Ecommerce FAQs",
        "Customer-facing ecommerce questions covering shipping, returns, and support.",
    ),
}


def load_dataset_specs(common_path: str) -> list[DatasetSpec]:
    """Load CSV files from a directory as dataset specifications."""
    specs: list[DatasetSpec] = []
    files = sorted(Path(common_path).glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {common_path}.")

    for file_path in files:
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to load '{file_path}' as a dataframe: {exc}"
            ) from exc

        file_name = file_path.name
        dataset_name, description = DATASET_METADATA.get(
            file_name,
            (file_path.stem, f"Dataset loaded from {file_name}."),
        )
        specs.append(
            DatasetSpec(
                name=dataset_name,
                dataframe=dataframe,
                description=description,
                source_path=str(file_path),
            )
        )
        print(
            f"SUCCESS: Loaded '{file_path}' as '{dataset_name}' with "
            f"{len(dataframe)} rows and {len(dataframe.columns)} columns"
        )
    return specs


def extract_assistant_text(agent_result: dict) -> str:
    """Extract the latest textual AI message from an agent result."""
    messages = agent_result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    return "\n".join(parts)
    return "The agent did not return a text response."


def main() -> None:
    """Run the interactive dataframe-agent session."""
    try:
        catalog = DataframeCatalog.from_specs(load_dataset_specs(COMMON_PATH))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    system_prompt = SystemMessage(
        content=(
            "You are a smart data assistant capable of reading multiple CSV files.\n"
            "Use tools before answering.\n"
            "The available datasets are SaaS Docs, Credit Card Terms, Hospital Policy, "
            "and Ecommerce FAQs.\n"
            "Do not answer from general knowledge.\n"
            "Answer in plain English and mention which dataset you used."
        )
    )

    model = ChatOpenAI(model="gpt-4o-mini")
    agent = create_pandas_like_agent(
        model=model,
        catalog=catalog,
        system_prompt=system_prompt,
        debug=True,
    )

    print("\nAI Agent is initialized and ready to answer questions!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        if not user_input.strip():
            continue
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            print(extract_assistant_text(response))
        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
