"""Result-set normalisation and comparison (shared by the verifier and the evaluation)."""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections import Counter
from decimal import Decimal


def _norm_val(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
        if math.isnan(f):
            return "nan"
        if f.is_integer() and abs(f) < 1e15:
            return int(f)
        return float(f"{f:.6g}")  # 6 significant digits: tolerant to float/decimal noise
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    return str(v)


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_val(v) for v in r) for r in rows]


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    """Execution-accuracy equality. Order matters only if `ordered` (gold has ORDER BY).

    Column order is ignored (a permutation of the predicted columns may match), duplicates count.
    """
    g, p = normalize_rows(gold), normalize_rows(pred)
    if len(g) != len(p):
        return False
    if not g:
        return True
    ng, np_ = len(g[0]), len(p[0])
    if ng != np_:
        return False
    perms = [tuple(range(ng))] if ng > 6 else itertools.permutations(range(ng))
    for perm in perms:
        pp = [tuple(r[i] for i in perm) for r in p]
        if (pp == g) if ordered else (Counter(pp) == Counter(g)):
            return True
    return False
