"""Pandera contract for every dataset split (template and final)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

SPLITS = ["train", "val", "test_in", "test_ood"]
KINDS = [
    "count_all",
    "count_where",
    "agg",
    "list_where",
    "topk",
    "group_agg",
    "group_count",
    "having",
    "count_distinct",
    "above_avg",
    "time_group",
    "in_subquery",
]

_base_columns = {
    "id": pa.Column(str, unique=True),
    "split": pa.Column(str, pa.Check.isin(SPLITS)),
    "db": pa.Column(str, pa.Check.isin(["tpch", "smard", "weather"])),
    "kind": pa.Column(str, pa.Check.isin(KINDS)),
    "lang": pa.Column(str, pa.Check.isin(["de", "en"])),
    "question_template": pa.Column(str, pa.Check.str_length(min_value=10)),
    "sql": pa.Column(str, [pa.Check.str_startswith("SELECT "), pa.Check.str_matches(r"^[^;]+$")]),
    "sql_orig": pa.Column(str, pa.Check.str_startswith("SELECT ")),
    "sql_hash": pa.Column(str, [pa.Check.str_matches(r"^[0-9a-f]{16}$")], unique=True),
    "n_rows": pa.Column(int, pa.Check.in_range(1, 10_000)),
    "ordered": pa.Column(bool),
    "renamed": pa.Column(bool),
    "variant": pa.Column(str),
    "ddl": pa.Column(str, pa.Check.str_startswith("CREATE TABLE ")),
    "num_joins": pa.Column(int, pa.Check.ge(0)),
    "num_tables": pa.Column(int, pa.Check.ge(1)),
}

TemplateSchema = pa.DataFrameSchema(_base_columns, strict=False)

FinalSchema = pa.DataFrameSchema(
    {
        **_base_columns,
        "question": pa.Column(str, pa.Check.str_length(min_value=10)),
        "question_source": pa.Column(str, pa.Check.isin(["template", "paraphrase"])),
    },
    strict=False,
)


def assert_no_leak(frames: dict[str, pl.DataFrame]) -> None:
    """No canonical SQL hash may appear in more than one split."""
    seen: dict[str, str] = {}
    for split, df in frames.items():
        for h in df["sql_hash"]:
            if h in seen:
                raise AssertionError(f"hash {h} in both {seen[h]} and {split}")
            seen[h] = split
