import unittest
from pathlib import Path

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_pandas import (
    DataframeCatalog,
    DatasetSpec,
    build_tools,
    create_pandas_like_agent,
)

DATA_DIR = Path("data/corpora/nautilus/misc")


class LangchainPandasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = DataframeCatalog.from_specs(
            [
                DatasetSpec(
                    name="Credit Card Terms",
                    dataframe=pd.read_csv(DATA_DIR / "credit_card_terms.csv"),
                    description="Credit card fees and APR terms.",
                    source_path=str(DATA_DIR / "credit_card_terms.csv"),
                ),
                DatasetSpec(
                    name="Ecommerce FAQs",
                    dataframe=pd.read_csv(DATA_DIR / "ecommerce_faqs.csv"),
                    description="Shipping and return FAQ entries.",
                    source_path=str(DATA_DIR / "ecommerce_faqs.csv"),
                ),
                DatasetSpec(
                    name="Hospital Policy",
                    dataframe=pd.read_csv(DATA_DIR / "hospital_policy.csv"),
                    description="Hospital operations and compliance policies.",
                    source_path=str(DATA_DIR / "hospital_policy.csv"),
                ),
                DatasetSpec(
                    name="SaaS Docs",
                    dataframe=pd.read_csv(DATA_DIR / "saas_docs.csv"),
                    description="Product documentation and API limits.",
                    source_path=str(DATA_DIR / "saas_docs.csv"),
                ),
            ]
        )
        cls.tools = {tool.name: tool for tool in build_tools(cls.catalog)}

    def test_catalog_infers_id_column_and_preserves_metadata(self) -> None:
        entry = self.catalog.get("Hospital Policy")

        self.assertEqual(entry.id_column, "Policy ID")
        self.assertEqual(entry.source_name, "hospital_policy.csv")
        self.assertEqual(
            entry.description, "Hospital operations and compliance policies."
        )

    def test_catalog_rejects_duplicate_names(self) -> None:
        dataframe = pd.DataFrame({"ID": [1]})

        with self.assertRaises(ValueError):
            DataframeCatalog.from_specs(
                [
                    DatasetSpec(name="Duplicate", dataframe=dataframe),
                    DatasetSpec(name="Duplicate", dataframe=dataframe),
                ]
            )

    def test_catalog_owns_a_defensive_dataframe_snapshot(self) -> None:
        dataframe = pd.DataFrame({"ID": [1], "Value": ["original"]})
        spec = DatasetSpec(name="Snapshot", dataframe=dataframe)
        catalog = DataframeCatalog.from_specs([spec])

        dataframe.loc[0, "Value"] = "input changed"
        spec_copy = spec.dataframe
        spec_copy.loc[0, "Value"] = "spec copy changed"
        entry_copy = catalog.get("Snapshot").dataframe
        entry_copy.loc[0, "Value"] = "entry copy changed"

        self.assertEqual(
            catalog.get("Snapshot").dataframe.loc[0, "Value"],
            "original",
        )

    def test_list_dataframes_reports_known_datasets(self) -> None:
        output = self.tools["list_dataframes"].invoke({})

        self.assertIn("SaaS Docs", output)
        self.assertIn("15 rows x 6 columns", output)
        self.assertIn("source=saas_docs.csv", output)

    def test_describe_dataframe_reports_expected_columns(self) -> None:
        output = self.tools["describe_dataframe"].invoke({"dataset_name": "SaaS Docs"})

        self.assertIn("Technical Limit", output)
        self.assertIn("Related API", output)
        self.assertIn("ID column: Doc ID", output)

    def test_search_rows_finds_expected_text_matches(self) -> None:
        output = self.tools["search_rows"].invoke(
            {"query": "electronics return policy"}
        )

        self.assertIn("Ecommerce FAQs", output)
        self.assertIn("What is the return policy for electronics?", output)

    def test_filter_rows_supports_contains_and_column_projection(self) -> None:
        output = self.tools["filter_rows"].invoke(
            {
                "dataset_name": "Hospital Policy",
                "conditions": [
                    {
                        "column": "Policy Text",
                        "operator": "contains",
                        "value": "visitors",
                    }
                ],
                "columns": ["Policy ID", "Topic"],
            }
        )

        self.assertIn("HP-001", output)
        self.assertIn("Visiting Hours", output)

    def test_filter_rows_raises_clear_error_for_unknown_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown columns: Missing"):
            self.tools["filter_rows"].invoke(
                {
                    "dataset_name": "SaaS Docs",
                    "conditions": [
                        {"column": "Missing", "operator": "eq", "value": "x"}
                    ],
                }
            )

    def test_aggregate_rows_computes_grouped_counts(self) -> None:
        output = self.tools["aggregate_rows"].invoke(
            {
                "dataset_name": "SaaS Docs",
                "group_by": ["Feature"],
                "metric": "count",
            }
        )

        self.assertIn("| Feature", output)
        self.assertIn("API Rate Limit", output)
        self.assertNotIn("|   index |", output)

    def test_distinct_values_counts_categories(self) -> None:
        output = self.tools["distinct_values"].invoke(
            {
                "dataset_name": "Credit Card Terms",
                "column": "Category",
                "limit": 5,
            }
        )

        self.assertIn("Annual Fee", output)
        self.assertIn("| Category", output)

    def test_agent_factory_builds_compiled_graph(self) -> None:
        agent = create_pandas_like_agent(
            FakeListChatModel(responses=["final answer"]),
            self.catalog,
        )

        self.assertEqual(agent.name, "pandas_like_agent")


if __name__ == "__main__":
    unittest.main()
