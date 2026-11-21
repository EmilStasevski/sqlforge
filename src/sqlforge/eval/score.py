"""Execution-based scoring + scripted error taxonomy (no manual labels)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import duckdb
import sqlnorm

from sqlforge.schemas.card import load_card
from sqlforge.schemas.variant import Variant
from sqlforge.verify.execute import QueryTimeout, run_query
from sqlforge.verify.results import results_match

_FENCE = re.compile(r"```(?:sql|duckdb)?\s*(.*?)```", re.S | re.I)
_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
ERROR_CLASSES = [
    "correct",
    "empty",
    "syntax",
    "not_select",
    "wrong_schema_ref",
    "exec_error",
    "timeout",
    "wrong_aggregate",
    "wrong_result",
]


def extract_sql(text: str) -> str:
    """First SQL statement in a model response (handles ``` fences and trailing prose)."""
    m = _FENCE.search(text)
    s = m.group(1) if m else text
    s = s.strip()
    i = s.upper().find("SELECT")
    j = s.upper().find("WITH")
    starts = [k for k in (i, j) if k >= 0]
    if not starts:
        return ""
    s = s[min(starts) :]
    return s.split(";")[0].strip()


def _aggs(sql: str) -> list[str]:
    return sorted(a.upper() for a in _AGG.findall(sql))


class Scorer:
    def __init__(self):
        self._cons: dict[str, duckdb.DuckDBPyConnection] = {}

    def con(self, variant_json: str) -> duckdb.DuckDBPyConnection:
        if variant_json not in self._cons:
            if len(self._cons) >= 32:
                for c in self._cons.values():
                    c.close()
                self._cons.clear()
            v = Variant.from_json(variant_json)
            self._cons[variant_json] = v.connect(str(load_card(v.db).db_path))
        return self._cons[variant_json]

    def score(self, pred_sql: str, gold_sql: str, variant_json: str, ordered: bool) -> str:
        if not pred_sql:
            return "empty"
        try:
            sqlnorm.normalize(pred_sql)
        except ValueError:
            return "syntax"
        if not sqlnorm.is_safe_select(pred_sql):
            return "not_select"
        con = self.con(variant_json)
        gold = run_query(con, gold_sql)
        try:
            pred = run_query(con, pred_sql)
        except QueryTimeout:
            return "timeout"
        except (duckdb.CatalogException, duckdb.BinderException):
            return "wrong_schema_ref"
        except duckdb.ParserException:
            return "syntax"
        except duckdb.Error:
            return "exec_error"
        if results_match(gold, pred, ordered):
            return "correct"
        return "wrong_aggregate" if _aggs(pred_sql) != _aggs(gold_sql) else "wrong_result"


def bucket_of(row: dict) -> dict[str, str]:
    j = row["num_joins"]
    return {
        "joins": "0" if j == 0 else ("1" if j == 1 else "2+"),
        "aggregation": "yes" if row["has_aggregate"] else "no",
        "group_by": "yes" if row["has_group_by"] else "no",
        "subquery": "yes" if row["has_subquery"] else "no",
        "lang": row["lang"],
        "db": row["db"],
        "kind": row["kind"],
        "question_source": row["question_source"],
    }


def summarize(rows: list[dict]) -> dict:
    """rows: dataset rows with an added 'outcome' key."""
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    executes = n - sum(
        outcomes[k] for k in ("empty", "syntax", "not_select", "wrong_schema_ref", "exec_error", "timeout")
    )
    selects = n - sum(outcomes[k] for k in ("empty", "syntax", "not_select"))
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        for dim, val in bucket_of(r).items():
            b = buckets[dim][val]
            b[0] += r["outcome"] == "correct"
            b[1] += 1
    return {
        "n": n,
        "execution_accuracy": round(outcomes["correct"] / n, 4),
        "valid_sql_rate": round(executes / n, 4),
        "select_only_rate": round(selects / n, 4),
        "error_taxonomy": {k: outcomes.get(k, 0) for k in ERROR_CLASSES},
        "buckets": {
            dim: {
                val: {"n": t, "execution_accuracy": round(c / t, 4)} for val, (c, t) in sorted(vals.items())
            }
            for dim, vals in buckets.items()
        },
    }
