"""Query specs and their SQL rendering under a schema variant.

A Spec is schema-level (original table/column names). `render_sql(spec, variant)` produces the
DuckDB SQL for any variant (renamed/subset schema); `nlgen` renders the question from the same spec.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from decimal import Decimal

from sqlforge.schemas.card import Column
from sqlforge.schemas.variant import Variant

AGGS = ("AVG", "SUM", "MIN", "MAX")


@dataclass(frozen=True)
class Sel:
    func: str | None = None  # None (plain column), COUNT, COUNT_DISTINCT, AVG, SUM, MIN, MAX, YEAR, MONTH
    col: Column | None = None  # None only for COUNT(*)
    round: int | None = None

    @property
    def is_agg(self) -> bool:
        return self.func in ("COUNT", "COUNT_DISTINCT", *AGGS)


@dataclass(frozen=True)
class Filter:
    col: Column
    op: str  # =, >, <, IN, BETWEEN, YEAR, MONTH, DAY, BOOL
    values: tuple = ()


@dataclass(frozen=True)
class SubFilter:
    kind: str  # "above_avg" | "in" | "not_in"
    col: Column  # outer column (num for above_avg, key for in/not_in)
    child: Column | None = None  # child fk column (in/not_in)
    child_filters: tuple[Filter, ...] = ()


@dataclass
class Spec:
    kind: str
    base: str
    joins: list[tuple[Column, Column]] = field(default_factory=list)  # m:1 edges, in join order
    select: list[Sel] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    sub: SubFilter | None = None
    group_by: list[int] = field(default_factory=list)  # indices into select
    having: tuple[str, int] | None = None  # (op, value) on COUNT(*)
    order_by: tuple[int, str] | None = None  # (select index, ASC|DESC)
    limit: int | None = None

    def columns(self) -> set[Column]:
        cols = {s.col for s in self.select if s.col is not None}
        cols |= {f.col for f in self.filters}
        for fk, pk in self.joins:
            cols |= {fk, pk}
        if self.sub:
            cols.add(self.sub.col)
            if self.sub.child:
                cols.add(self.sub.child)
            cols |= {f.col for f in self.sub.child_filters}
        return cols

    def tables(self) -> set[str]:
        return {self.base} | {c.table for c in self.columns()}

    def with_limit(self, k: int | None) -> Spec:
        return replace(self, limit=k)


# ---------------------------------------------------------------- literals
def sql_literal(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dt.datetime):
        return f"TIMESTAMP '{v:%Y-%m-%d %H:%M:%S}'"
    if isinstance(v, dt.date):
        return f"DATE '{v:%Y-%m-%d}'"
    if isinstance(v, (int, float, Decimal)):
        return fmt_num(v)
    return "'" + str(v).replace("'", "''") + "'"


def fmt_num(v) -> str:
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    return f"{float(v):g}" if abs(float(v)) < 1e6 else str(v)


# ---------------------------------------------------------------- rendering
class _R:
    def __init__(self, v: Variant, qualify: bool):
        self.v, self.qualify = v, qualify

    def col(self, c: Column, qualify: bool | None = None) -> str:
        q = self.qualify if qualify is None else qualify
        name = self.v.c(c.table, c.name)
        return f"{self.v.t(c.table)}.{name}" if q else name

    def sel(self, s: Sel) -> str:
        if s.func == "COUNT":
            e = "COUNT(*)" if s.col is None else f"COUNT({self.col(s.col)})"
        elif s.func == "COUNT_DISTINCT":
            e = f"COUNT(DISTINCT {self.col(s.col)})"
        elif s.func in AGGS:
            e = f"{s.func}({self.col(s.col)})"
        elif s.func == "YEAR":
            return f"year({self.col(s.col)}) AS year"
        elif s.func == "MONTH":
            return f"date_trunc('month', {self.col(s.col)}) AS month"
        else:
            e = self.col(s.col)
        return f"ROUND({e}, {s.round})" if s.round is not None else e

    def ref(self, s: Sel) -> str:
        """Expression usable in GROUP BY / ORDER BY."""
        if s.func == "YEAR":
            return "year"
        if s.func == "MONTH":
            return "month"
        return self.sel(s)

    def filt(self, f: Filter, qualify: bool | None = None) -> str:
        c = self.col(f.col, qualify)
        if f.op in ("=", ">", "<", ">=", "<=", "!="):
            return f"{c} {f.op} {sql_literal(f.values[0])}"
        if f.op == "IN":
            return f"{c} IN ({', '.join(sql_literal(x) for x in f.values)})"
        if f.op == "BETWEEN":
            return f"{c} BETWEEN {sql_literal(f.values[0])} AND {sql_literal(f.values[1])}"
        if f.op == "YEAR":
            return f"year({c}) = {f.values[0]}"
        if f.op == "MONTH":  # values = (first day, first day of next month)
            lo, hi = f.values
            if f.col.role == "ts":
                lo, hi = dt.datetime.combine(lo, dt.time()), dt.datetime.combine(hi, dt.time())
            return f"{c} >= {sql_literal(lo)} AND {c} < {sql_literal(hi)}"
        if f.op == "DAY":
            if f.col.role == "ts":
                return f"CAST({c} AS DATE) = {sql_literal(f.values[0])}"
            return f"{c} = {sql_literal(f.values[0])}"
        if f.op == "BOOL":
            return c if f.values[0] else f"NOT {c}"
        raise ValueError(f.op)


def render_sql(spec: Spec, v: Variant) -> str:
    r = _R(v, qualify=bool(spec.joins) or (spec.sub is not None and spec.sub.kind != "above_avg"))
    parts = ["SELECT " + ", ".join(r.sel(s) for s in spec.select)]
    frm = f"FROM {v.t(spec.base)}"
    for fk, pk in spec.joins:
        frm += f" JOIN {v.t(pk.table)} ON {r.col(fk, True)} = {r.col(pk, True)}"
    parts.append(frm)
    conds = [r.filt(f) for f in spec.filters]
    if spec.sub:
        s = spec.sub
        if s.kind == "above_avg":
            conds.append(f"{r.col(s.col)} > (SELECT AVG({r.col(s.col)}) FROM {v.t(s.col.table)})")
        else:
            inner = f"SELECT {r.col(s.child, True)} FROM {v.t(s.child.table)}"
            if s.child_filters:
                inner += " WHERE " + " AND ".join(r.filt(f, True) for f in s.child_filters)
            neg = "NOT IN" if s.kind == "not_in" else "IN"
            conds.append(f"{r.col(s.col)} {neg} ({inner})")
    if conds:
        parts.append("WHERE " + " AND ".join(conds))
    if spec.group_by:
        parts.append("GROUP BY " + ", ".join(r.ref(spec.select[i]) for i in spec.group_by))
    if spec.having:
        parts.append(f"HAVING COUNT(*) {spec.having[0]} {spec.having[1]}")
    if spec.order_by:
        i, d = spec.order_by
        parts.append(f"ORDER BY {r.ref(spec.select[i])}" + (" DESC" if d == "DESC" else ""))
    if spec.limit is not None:
        parts.append(f"LIMIT {spec.limit}")
    return " ".join(parts)
