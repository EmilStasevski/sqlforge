"""Schema variants (augmentation): table/column subsets, shuffled order, renamed identifiers.

A variant is serialisable (`to_json`) and fully determines (a) the DDL shown in the prompt and
(b) the execution environment: an in-memory DuckDB with the source DB attached READ_ONLY and one
view per variant table that exposes only the variant's columns under the variant's names.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import duckdb

from sqlforge.schemas.card import Card, Column

# Renamed identifiers must not collide with SQL keywords (would need quoting).
_RESERVED = {
    "date",
    "hour",
    "year",
    "month",
    "day",
    "order",
    "group",
    "select",
    "from",
    "where",
    "table",
    "name",
    "type",
    "size",
    "time",
    "timestamp",
    "key",
    "value",
    "values",
    "user",
    "level",
    "position",
    "count",
    "sum",
    "min",
    "max",
    "avg",
    "limit",
    "offset",
    "default",
    "end",
    "start",
    "range",
    "all",
    "any",
}


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _strip_prefix(name: str) -> str:
    m = re.match(r"^[a-z]{1,2}_(.+)$", name)  # TPC-H style c_acctbal -> acctbal
    return m.group(1) if m else name


@dataclass
class VTable:
    orig: str
    name: str
    columns: list[tuple[str, str, str]]  # (orig, new, type)

    def col(self, orig: str) -> str:
        for o, n, _ in self.columns:
            if o == orig:
                return n
        raise KeyError(f"{self.orig}.{orig} not in variant")


@dataclass
class Variant:
    db: str
    tables: list[VTable]
    fks: list[tuple[str, str, str, str]]  # (table, col, ref_table, ref_col), variant names

    def t(self, orig: str) -> str:
        return self._vt(orig).name

    def c(self, table: str, col: str) -> str:
        return self._vt(table).col(col)

    def _vt(self, orig: str) -> VTable:
        for vt in self.tables:
            if vt.orig == orig:
                return vt
        raise KeyError(f"table {orig} not in variant")

    def has(self, table: str, col: str | None = None) -> bool:
        try:
            vt = self._vt(table)
            return col is None or any(o == col for o, _, _ in vt.columns)
        except KeyError:
            return False

    @property
    def is_identity(self) -> bool:
        return all(vt.orig == vt.name and all(o == n for o, n, _ in vt.columns) for vt in self.tables)

    def ddl(self) -> str:
        lines = []
        for vt in self.tables:
            cols = []
            for _, n, typ in vt.columns:
                ref = next(
                    (f" REFERENCES {rt}({rc})" for t, c, rt, rc in self.fks if t == vt.name and c == n), ""
                )
                cols.append(f"{n} {typ}{ref}")
            lines.append(f"CREATE TABLE {vt.name} ({', '.join(cols)});")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "db": self.db,
                "tables": [[vt.orig, vt.name, vt.columns] for vt in self.tables],
                "fks": self.fks,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(s: str) -> Variant:
        d = json.loads(s)
        return Variant(
            d["db"],
            [VTable(o, n, [tuple(c) for c in cols]) for o, n, cols in d["tables"]],
            [tuple(f) for f in d["fks"]],
        )

    def connect(self, db_path: str) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        con.execute("SET threads = 2")
        con.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
        for vt in self.tables:
            cols = ", ".join(f'"{o}" AS "{n}"' for o, n, _ in vt.columns)
            con.execute(f'CREATE VIEW "{vt.name}" AS SELECT {cols} FROM src.main."{vt.orig}"')
        return con


def full_variant(card: Card) -> Variant:
    """All tables and columns, original names and order (used for val/test and serving)."""
    tables = [
        VTable(t.name, t.name, [(c.name, c.name, c.type) for c in t.columns]) for t in card.tables.values()
    ]
    fks = [(fk.table, fk.name, pk.table, pk.name) for fk, pk in card.joins]
    return Variant(card.db, tables, fks)


def random_variant(
    card: Card,
    required: set[Column],
    rng: random.Random,
    required_tables: set[str] = frozenset(),
    p_extra_table: float = 0.4,
    p_extra_col: float = 0.7,
    p_shuffle: float = 0.5,
    p_rename: float = 0.3,
) -> Variant:
    """Subschema containing at least `required` columns; optionally shuffled and renamed."""
    req_tables = {c.table for c in required} | set(required_tables)
    tables = [t for t in card.tables.values() if t.name in req_tables or rng.random() < p_extra_table]
    if rng.random() < p_shuffle:
        rng.shuffle(tables)
    rename = rng.random() < p_rename
    used_tnames: set[str] = set()
    vts: list[VTable] = []
    for t in tables:
        cols = [c for c in t.columns if c in required or c.role in ("key",) or rng.random() < p_extra_col]
        if rng.random() < p_shuffle:
            rng.shuffle(cols)
        tname = t.name
        if rename:
            cands = [_snake(t.en[1]), _snake(t.en[0]), f"{t.name}_tbl"]
            cands = [x for x in cands if x not in _RESERVED and x not in used_tnames and x != t.name]
            tname = rng.choice(cands) if cands else t.name
        used_tnames.add(tname)
        used: set[str] = set()
        vcols = []
        for c in cols:
            new = c.name
            if rename:
                cands = [_strip_prefix(c.name)]
                if c.en:
                    cands.append(_snake(c.en))
                cands = [x for x in cands if x and x not in _RESERVED and x not in used and x[0].isalpha()]
                new = rng.choice(cands) if cands else c.name
                if new in used:
                    new = c.name
            used.add(new)
            vcols.append((c.name, new, c.type))
        vts.append(VTable(t.name, tname, vcols))
    v = Variant(card.db, vts, [])
    for fk, pk in card.joins:
        if v.has(fk.table, fk.name) and v.has(pk.table, pk.name):
            v.fks.append((v.t(fk.table), v.c(fk.table, fk.name), v.t(pk.table), v.c(pk.table, pk.name)))
    return v
