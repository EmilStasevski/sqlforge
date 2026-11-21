"""Execution verifier: the automatic 'no manual check' guarantee for SQL labels.

A candidate is kept only if all checks pass:
  safe     single read-only SELECT (Rust sqlnorm)
  exec     executes on a read-only DuckDB within the timeout
  rows     1 <= rows <= 10_000 and not all-NULL
  limit    LIMIT implies ORDER BY
  determ   same result single-threaded vs multi-threaded
  ties     ORDER BY key has no ties among the returned rows (incl. the row after LIMIT)
  variant  the renamed/subset-schema SQL (the training target) returns the same result
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import sqlnorm

from sqlforge.sampler.spec import Spec, render_sql
from sqlforge.schemas.card import Card
from sqlforge.schemas.variant import Variant, full_variant
from sqlforge.verify.results import normalize_rows, results_match

MAX_ROWS = 10_000
TIMEOUT_S = 5.0


class QueryTimeout(Exception):
    pass


def run_query(
    con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = TIMEOUT_S, max_rows: int = MAX_ROWS + 1
) -> list[tuple]:
    """Execute with a wall-clock timeout (DuckDB interrupt) and a row cap."""
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        return cur.fetchmany(max_rows)
    except duckdb.InterruptException as e:
        raise QueryTimeout(sql) from e
    finally:
        timer.cancel()


@dataclass
class Verdict:
    ok: bool
    reason: str
    n_rows: int = 0


class Verifier:
    def __init__(self, card: Card):
        self.card = card
        self.full = full_variant(card)
        # Few threads per connection: the builder runs one Verifier per worker process.
        self.con = duckdb.connect(str(card.db_path), read_only=True)
        self.con.execute("SET threads = 2")
        self.con1 = duckdb.connect(str(card.db_path), read_only=True)
        self.con1.execute("SET threads = 1")
        self._variant_cons: dict[str, duckdb.DuckDBPyConnection] = {}

    def variant_con(self, v: Variant) -> duckdb.DuckDBPyConnection:
        key = v.to_json()
        if key not in self._variant_cons:
            if len(self._variant_cons) >= 8:
                for c in self._variant_cons.values():
                    c.close()
                self._variant_cons.clear()
            self._variant_cons[key] = v.connect(str(self.card.db_path))
        return self._variant_cons[key]

    def check(self, spec: Spec, variant: Variant) -> Verdict:
        sql = render_sql(spec, self.full)
        target = render_sql(spec, variant)
        for s in {sql, target}:
            if not sqlnorm.is_safe_select(s):
                return Verdict(False, "safe")
        feats = sqlnorm.features(sql)
        if feats["has_limit"] and not feats["has_order_by"]:
            return Verdict(False, "limit")
        try:
            rows = run_query(self.con, sql)
        except QueryTimeout:
            return Verdict(False, "timeout")
        except duckdb.Error:
            return Verdict(False, "exec")
        if not 1 <= len(rows) <= MAX_ROWS:
            return Verdict(False, "rows", len(rows))
        if all(v is None for r in rows for v in r):
            return Verdict(False, "null", len(rows))
        ordered = spec.order_by is not None
        try:
            rows1 = run_query(self.con1, sql)
        except (QueryTimeout, duckdb.Error):
            return Verdict(False, "determ")
        if not results_match(rows, rows1, ordered):
            return Verdict(False, "determ")
        if spec.order_by is not None:
            idx = spec.order_by[0]
            check_rows = rows
            if spec.limit is not None:
                check_rows = run_query(self.con, render_sql(spec.with_limit(spec.limit + 1), self.full))
            keys = [r[idx] for r in normalize_rows(check_rows)]
            if len(set(keys)) != len(keys):
                return Verdict(False, "ties")
        if target != sql:
            try:
                vrows = run_query(self.variant_con(variant), target)
            except (QueryTimeout, duckdb.Error):
                return Verdict(False, "variant")
            if not results_match(rows, vrows, ordered):
                return Verdict(False, "variant")
        return Verdict(True, "ok", len(rows))
