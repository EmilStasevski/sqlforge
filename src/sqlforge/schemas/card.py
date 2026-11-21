"""Schema cards: YAML glossary/roles + types introspected from DuckDB + cached value statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import duckdb
import yaml

CARD_DIR = Path(__file__).parent / "cards"
DB_DIR = Path("data/db")


@dataclass(frozen=True)
class Column:
    table: str
    name: str
    type: str
    role: str | None = None  # key, fk, name, cat, num, date, ts, bool; None = unused by the sampler
    en: str | None = None
    de: str | None = None


@dataclass
class Table:
    name: str
    en: tuple[str, str]
    de: tuple[str, str]
    columns: list[Column]

    def col(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)

    def by_role(self, *roles: str) -> list[Column]:
        return [c for c in self.columns if c.role in roles]


@dataclass
class Card:
    db: str
    tables: dict[str, Table]
    joins: list[tuple[Column, Column]]  # (fk side, pk side): many-to-one
    db_path: Path
    _stats: dict = field(default_factory=dict, repr=False)

    def col(self, table: str, name: str) -> Column:
        return self.tables[table].col(name)

    def parents(self, table: str) -> list[tuple[Column, Column]]:
        """Many-to-one edges leaving `table`."""
        return [(fk, pk) for fk, pk in self.joins if fk.table == table]

    def children(self, table: str) -> list[tuple[Column, Column]]:
        """One-to-many edges arriving at `table` (fk side is the child)."""
        return [(fk, pk) for fk, pk in self.joins if pk.table == table]

    def m1_paths(self, base: str, max_hops: int = 2) -> list[list[tuple[Column, Column]]]:
        """All many-to-one join chains starting at `base` (no row duplication)."""
        out: list[list[tuple[Column, Column]]] = [[]]
        frontier: list[list[tuple[Column, Column]]] = [[]]
        for _ in range(max_hops):
            nxt = []
            for path in frontier:
                here = path[-1][1].table if path else base
                seen = {base, *(e[1].table for e in path)}
                for fk, pk in self.parents(here):
                    if pk.table not in seen:
                        nxt.append([*path, (fk, pk)])
            out += nxt
            frontier = nxt
        return out

    # ---- value statistics (computed once per process, from the real data) ----
    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    def distinct_values(self, c: Column, limit: int = 300) -> list:
        key = ("distinct", c.table, c.name)
        if key not in self._stats:
            with self.connect() as con:
                rows = con.execute(
                    f'SELECT DISTINCT "{c.name}" FROM "{c.table}" WHERE "{c.name}" IS NOT NULL '
                    f'ORDER BY hash("{c.name}") LIMIT {limit}'
                ).fetchall()
            self._stats[key] = [r[0] for r in rows]
        return self._stats[key]

    def n_distinct(self, c: Column) -> int:
        key = ("ndistinct", c.table, c.name)
        if key not in self._stats:
            with self.connect() as con:
                self._stats[key] = con.execute(
                    f'SELECT COUNT(DISTINCT "{c.name}") FROM "{c.table}"'
                ).fetchone()[0]
        return self._stats[key]

    def n_rows(self, table: str) -> int:
        key = ("nrows", table)
        if key not in self._stats:
            with self.connect() as con:
                self._stats[key] = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        return self._stats[key]

    def quantiles(self, c: Column) -> list[float]:
        key = ("quant", c.table, c.name)
        if key not in self._stats:
            with self.connect() as con:
                probs = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]"
                q = con.execute(f'SELECT quantile_cont("{c.name}", {probs}) FROM "{c.table}"').fetchone()[0]
            self._stats[key] = [float(x) for x in q]
        return self._stats[key]

    def time_range(self, c: Column) -> tuple:
        key = ("range", c.table, c.name)
        if key not in self._stats:
            with self.connect() as con:
                self._stats[key] = con.execute(
                    f'SELECT min("{c.name}"), max("{c.name}") FROM "{c.table}"'
                ).fetchone()
        return self._stats[key]


def load_card(db: str) -> Card:
    return _load_card(db, DB_DIR)  # DB_DIR read at call time (tests point it at fixtures)


@cache
def _load_card(db: str, db_dir: Path) -> Card:
    raw = yaml.safe_load((CARD_DIR / f"{db}.yaml").read_text())
    db_path = db_dir / f"{db}.duckdb"
    con = duckdb.connect(str(db_path), read_only=True)
    types = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    con.close()
    tables: dict[str, Table] = {}
    for tname, tdef in raw["tables"].items():
        cdefs = tdef.get("columns", {})
        cols = [Column(tname, cn, ct, **cdefs.get(cn, {})) for t, cn, ct in types if t == tname]
        if not cols:
            raise ValueError(f"{db}: table {tname} from card not found in {db_path}")
        missing = set(cdefs) - {c.name for c in cols}
        if missing:
            raise ValueError(f"{db}.{tname}: card columns not in database: {missing}")
        tables[tname] = Table(tname, tuple(tdef["en"]), tuple(tdef["de"]), cols)
    joins = []
    for a, b in raw.get("joins", []):
        (ta, ca), (tb, cb) = a.split("."), b.split(".")
        joins.append((tables[ta].col(ca), tables[tb].col(cb)))
    return Card(db, tables, joins, db_path)
