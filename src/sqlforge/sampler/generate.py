"""Grammar-based random spec generator. Literals are drawn from the real data (DuckDB statistics)."""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable

from sqlforge.sampler.spec import AGGS, Filter, Sel, Spec, SubFilter
from sqlforge.schemas.card import Card, Column

MAX_GROUPS = 100

KIND_WEIGHTS: dict[str, float] = {
    "count_all": 0.3,
    "count_where": 1.2,
    "agg": 1.5,
    "list_where": 1.2,
    "topk": 1.2,
    "group_agg": 1.6,
    "group_count": 1.0,
    "having": 0.6,
    "count_distinct": 0.6,
    "above_avg": 0.6,
    "time_group": 1.0,
    "in_subquery": 0.7,
}


def is_catlike(c: Column) -> bool:
    return c.role in ("cat", "name") or (c.role in ("fk", "key") and c.type == "VARCHAR")


def is_label(c: Column) -> bool:
    return c.role in ("name", "key") or is_catlike(c)


def _round_num(c: Column, x: float):
    if c.type in ("INTEGER", "BIGINT"):
        return int(round(x))
    if abs(x) >= 100:
        return int(round(x))
    if abs(x) >= 1:
        return round(x, 1)
    return round(x, 2)


class Sampler:
    def __init__(self, card: Card, rng: random.Random):
        self.card, self.rng = card, rng

    # ------------------------------------------------------------ helpers
    def _reachable(self, base: str, pred: Callable[[Column], bool], allow_join: bool = True):
        """(path, column) pairs: column satisfies pred and lives on base or an m:1-reachable table."""
        out = []
        paths = self.card.m1_paths(base) if allow_join else [[]]
        for path in paths:
            t = path[-1][1].table if path else base
            for c in self.card.tables[t].columns:
                if pred(c) and not (path and c == path[-1][1]):
                    out.append((path, c))
        return out

    def _pick(self, items):
        return self.rng.choice(items) if items else None

    def _literal_filter(self, c: Column) -> Filter | None:
        rng, card = self.rng, self.card
        if is_catlike(c):
            vals = card.distinct_values(c)
            if not vals:
                return None
            if len(vals) >= 3 and c.role == "cat" and rng.random() < 0.2:
                return Filter(c, "IN", tuple(sorted(rng.sample(vals, 2), key=str)))
            return Filter(c, "=", (rng.choice(vals),))
        if c.role == "num":
            q = card.quantiles(c)
            if rng.random() < 0.15:
                lo, hi = sorted(rng.sample(q, 2))
                lo, hi = _round_num(c, lo), _round_num(c, hi)
                if lo < hi:
                    return Filter(c, "BETWEEN", (lo, hi))
            return Filter(c, rng.choice([">", "<"]), (_round_num(c, rng.choice(q[1:-1])),))
        if c.role in ("date", "ts"):
            lo, hi = card.time_range(c)
            lo_d = lo.date() if isinstance(lo, dt.datetime) else lo
            hi_d = hi.date() if isinstance(hi, dt.datetime) else hi
            mode = rng.choice(["YEAR", "MONTH", "MONTH", "DAY", ">=", "<"])
            if mode == "YEAR":
                return Filter(c, "YEAR", (rng.randint(lo_d.year, hi_d.year),))
            day = lo_d + dt.timedelta(days=rng.randint(0, max(0, (hi_d - lo_d).days - 1)))
            if mode == "MONTH":
                first = day.replace(day=1)
                nxt = (first + dt.timedelta(days=32)).replace(day=1)
                return Filter(c, "MONTH", (first, nxt))
            if mode == "DAY":
                return Filter(c, "DAY", (day,))
            if c.role == "ts":
                return Filter(c, mode, (dt.datetime.combine(day, dt.time()),))
            return Filter(c, mode, (day,))
        if c.role == "bool":
            return Filter(c, "BOOL", (rng.random() < 0.5,))
        return None

    def _filter(self, base: str, allow_join: bool = True, exclude: set[Column] | None = None):
        exclude = exclude or set()
        cands = self._reachable(
            base,
            lambda c: (is_catlike(c) or c.role in ("num", "date", "ts", "bool")) and c not in exclude,
            allow_join,
        )
        pick = self._pick(cands)
        if pick is None:
            return [], None
        path, c = pick
        f = self._literal_filter(c)
        return (path, f) if f else ([], None)

    def _maybe_filter(self, spec: Spec, p: float = 0.5, allow_join: bool = True):
        if self.rng.random() < p:
            path, f = self._filter(spec.base, allow_join)
            if f:
                spec.filters.append(f)
                _merge_path(spec, path)

    def _base(self, pred: Callable[[Card, str], bool]) -> str | None:
        return self._pick([t for t in self.card.tables if pred(self.card, t)])

    def _label(self, base: str) -> Column | None:
        t = self.card.tables[base]
        return (
            self._pick(t.by_role("name"))
            or self._pick(t.by_role("key"))
            or self._pick([c for c in t.columns if is_label(c) or c.role == "ts"])
        )

    def _round(self, func: str) -> int | None:
        return self.rng.choice([1, 2]) if func == "AVG" and self.rng.random() < 0.4 else None

    # ------------------------------------------------------------ kinds
    def count_all(self):
        return Spec("count_all", self._pick(list(self.card.tables)), select=[Sel("COUNT")])

    def count_where(self):
        s = Spec("count_where", self._pick(list(self.card.tables)), select=[Sel("COUNT")])
        self._maybe_filter(s, 1.0)
        return s if s.filters else None

    def agg(self):
        base = self._base(lambda c, t: bool(c.tables[t].by_role("num")))
        num = self._pick(self.card.tables[base].by_role("num"))
        func = self.rng.choice(AGGS)
        s = Spec("agg", base, select=[Sel(func, num, self._round(func))])
        self._maybe_filter(s, 0.6)
        return s

    def list_where(self):
        base = self._pick(list(self.card.tables))
        t = self.card.tables[base]
        pool = [c for c in t.columns if c.role in ("name", "key", "cat", "num", "date", "ts")]
        if not pool:
            return None
        cols = self.rng.sample(pool, k=min(len(pool), self.rng.choice([1, 2, 2])))
        s = Spec("list_where", base, select=[Sel(None, c) for c in cols])
        self._maybe_filter(s, 1.0)
        if not s.filters:
            return None
        if self.rng.random() < 0.4:
            i = self.rng.randrange(len(cols))
            s.order_by, s.limit = (i, self.rng.choice(["ASC", "DESC"])), self.rng.choice([3, 5, 10])
        return s

    def topk(self):
        base = self._base(lambda c, t: bool(c.tables[t].by_role("num")))
        num = self._pick(self.card.tables[base].by_role("num"))
        label = self._label(base)
        if label is None:
            return None
        s = Spec("topk", base, select=[Sel(None, label), Sel(None, num)])
        s.order_by, s.limit = (1, self.rng.choice(["DESC", "DESC", "ASC"])), self.rng.choice([1, 3, 5, 10])
        self._maybe_filter(s, 0.4)
        return s

    def _groupable(self, c: Column, base: str) -> bool:
        """Low-cardinality column; on the base table it must not be unique (else every group has 1 row)."""
        if not (is_catlike(c) or c.role in ("bool", "fk")):
            return False
        n = self.card.n_distinct(c)
        return n <= MAX_GROUPS and (c.table != base or n < self.card.n_rows(base))

    def _group_col(self, base: str):
        return self._pick(self._reachable(base, lambda c: self._groupable(c, base)))

    def group_agg(self):
        base = self._base(lambda c, t: bool(c.tables[t].by_role("num")))
        g = self._group_col(base)
        if g is None:
            return None
        path, gcol = g
        num = self._pick(self.card.tables[base].by_role("num"))
        func = self.rng.choice(AGGS)
        s = Spec("group_agg", base, select=[Sel(None, gcol), Sel(func, num, self._round(func))], group_by=[0])
        _merge_path(s, path)
        self._maybe_filter(s, 0.35)
        if self.rng.random() < 0.35:
            s.order_by, s.limit = (1, "DESC"), self.rng.choice([1, 3, 5])
        return s

    def group_count(self):
        base = self._pick(list(self.card.tables))
        g = self._group_col(base)
        if g is None:
            return None
        path, gcol = g
        s = Spec("group_count", base, select=[Sel(None, gcol), Sel("COUNT")], group_by=[0])
        _merge_path(s, path)
        self._maybe_filter(s, 0.3)
        if self.rng.random() < 0.3:
            s.order_by, s.limit = (1, "DESC"), self.rng.choice([1, 3, 5])
        return s

    def having(self):
        base = self._pick(list(self.card.tables))
        gcol = self._pick([c for c in self.card.tables[base].columns if self._groupable(c, base)])
        if gcol is None:
            return None
        with self.card.connect() as con:
            q = con.execute(
                f'SELECT quantile_disc(n, [0.3, 0.5, 0.7]) FROM (SELECT COUNT(*) n FROM "{base}" '
                f'GROUP BY "{gcol.name}")'
            ).fetchone()[0]
        if not q:
            return None
        thr = int(self.rng.choice(q))
        s = Spec(
            "having", base, select=[Sel(None, gcol)], group_by=[0], having=(self.rng.choice([">", "<"]), thr)
        )
        self._maybe_filter(s, 0.5)
        if any(f.col == gcol for f in s.filters):
            return None
        return s

    def count_distinct(self):
        base = self._pick(list(self.card.tables))
        c = self._pick([c for c in self.card.tables[base].columns if is_catlike(c) or c.role == "fk"])
        if c is None:
            return None
        s = Spec("count_distinct", base, select=[Sel("COUNT_DISTINCT", c)])
        self._maybe_filter(s, 0.5, allow_join=False)
        return s

    def above_avg(self):
        base = self._base(lambda c, t: bool(c.tables[t].by_role("num")))
        num = self._pick(self.card.tables[base].by_role("num"))
        label = self._label(base)
        if label is None or label == num:
            return None
        s = Spec(
            "above_avg", base, select=[Sel(None, label), Sel(None, num)], sub=SubFilter("above_avg", num)
        )
        self._maybe_filter(s, 0.6, allow_join=False)
        if any(f.col == num for f in s.filters):
            return None
        if self.rng.random() < 0.6:
            s.order_by, s.limit = (1, "DESC"), self.rng.choice([3, 5, 10])
        return s

    def time_group(self):
        base = self._base(lambda c, t: bool(c.tables[t].by_role("date", "ts")))
        if base is None:
            return None
        t = self.card.tables[base]
        tcol = self._pick(t.by_role("date", "ts"))
        nums = t.by_role("num")
        if nums and self.rng.random() < 0.75:
            func = self.rng.choice(AGGS)
            m = Sel(func, self._pick(nums), self._round(func))
        else:
            m = Sel("COUNT")
        s = Spec("time_group", base, select=[Sel(self.rng.choice(["YEAR", "MONTH"]), tcol), m], group_by=[0])
        s.order_by = (0, "ASC")
        self._maybe_filter(s, 0.5)
        return s

    def in_subquery(self):
        parents = [t for t in self.card.tables if self.card.children(t)]
        if not parents:
            return None
        base = self._pick(parents)
        fk, pk = self._pick(self.card.children(base))
        label = self._label(base)
        if label is None:
            return None
        _, f = self._filter(fk.table, allow_join=False, exclude={fk})
        if f is None:
            return None
        kind = "not_in" if self.rng.random() < 0.25 else "in"
        s = Spec("in_subquery", base, select=[Sel(None, label)], sub=SubFilter(kind, pk, fk, (f,)))
        if self.rng.random() < 0.3:
            pf = self._literal_filter(
                self._pick([c for c in self.card.tables[base].columns if is_catlike(c)] or [label])
            )
            if pf and pf.col != label:
                s.filters.append(pf)
        return s

    # ------------------------------------------------------------ entry
    def sample(self, kind: str | None = None) -> Spec | None:
        kind = kind or self.rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
        return getattr(self, kind)()


def _merge_path(spec: Spec, path: list[tuple[Column, Column]]) -> None:
    for edge in path:
        if edge not in spec.joins:
            spec.joins.append(edge)
