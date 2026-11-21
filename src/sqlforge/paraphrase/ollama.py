"""Local-LLM paraphrasing (Ollama) + automatic preservation filter.

A paraphrase replaces the template question only if it passes `check_paraphrase`:
  - every literal (values, numbers, month names) appears verbatim
  - every glossary term appears (case-insensitive)
  - language unchanged (stop-word heuristic), single line, sane length, no SQL leaked into it
Otherwise the template question is kept. Results are cached (JSONL) so re-runs are cheap and stable.

Usage: uv run python -m sqlforge.paraphrase.ollama [--model M] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

from sqlforge.dataset.schema import FinalSchema, assert_no_leak

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
DATA_DIR = Path("data/dataset")
CACHE = Path("data/paraphrase_cache.jsonl")
SPLITS = ["train", "val", "test_in", "test_ood"]

_DE = {
    "der",
    "die",
    "das",
    "und",
    "ist",
    "sind",
    "von",
    "mit",
    "für",
    "pro",
    "wie",
    "viele",
    "gibt",
    "es",
    "deren",
    "welche",
    "zeige",
    "alle",
    "aller",
    "den",
    "dem",
    "des",
    "im",
    "nach",
    "als",
    "liegt",
    "auf",
}
_EN = {
    "the",
    "and",
    "is",
    "are",
    "of",
    "with",
    "for",
    "per",
    "how",
    "many",
    "there",
    "whose",
    "which",
    "show",
    "all",
    "what",
    "in",
    "than",
    "by",
    "each",
    "list",
    "have",
    "has",
}
_SQL = re.compile(r"\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY)\b")

# Operator concepts: if a trigger occurs in the template, the paraphrase must contain one of the synonyms.
# (lang, triggers in template, accepted synonyms in paraphrase); matching is lower-case substring.
CONCEPTS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("en", ("average",), ("average", "mean")),
    ("en", ("total",), ("total", "sum")),
    ("en", ("minimum",), ("minimum", "lowest", "smallest", "min ")),
    ("en", ("maximum",), ("maximum", "highest", "largest", "max ")),
    ("en", ("how many", "count", "number of"), ("how many", "count", "number")),
    ("en", ("distinct", "different"), ("distinct", "different", "unique")),
    (
        "en",
        ("greater than", "more than"),
        ("greater", "more than", "above", "over", "exceed", "higher than", "larger than"),
    ),
    ("en", ("less than", "fewer than"), ("less", "fewer", "below", "under", "lower than", "smaller than")),
    ("en", ("between",), ("between",)),
    ("en", ("before",), ("before", "prior to", "earlier than")),
    ("en", ("on or after",), ("on or after", "from", "since", "starting")),
    ("en", ("highest", "most"), ("highest", "most", "top", "largest", "greatest", "biggest", "maximum")),
    ("en", ("lowest",), ("lowest", "smallest", "least", "minimum", "bottom")),
    ("en", ("descending",), ("descending", "decreasing", "highest first", "largest first")),
    ("en", ("ascending",), ("ascending", "increasing", "lowest first", "smallest first", "earliest first")),
    ("en", ("chronolog",), ("chronolog", "sorted", "ordered", "in order")),
    ("en", (" no ", " not "), (" no ", " not ", "without", "none", "n't")),
    ("en", ("round",), ("round",)),
    ("en", ("first day of",), ("first day", "start of", "beginning of", "truncat")),
    (
        "en",
        ("one row per",),
        ("row", "per year", "per month", "each year", "each month", "every year", "every month"),
    ),
    ("en", ("per ", "each "), ("per ", "each ", "every ", " by ", "for all")),
    ("de", ("durchschnitt",), ("durchschnitt", "mittelwert", "mittlere")),
    ("de", ("summe",), ("summe", "gesamt", "insgesamt", "summiert")),
    ("de", ("minimum",), ("minimum", "minimal", "niedrigst", "kleinst", "geringst")),
    ("de", ("maximum",), ("maximum", "maximal", "höchst", "größt")),
    ("de", ("wie viele", "zähle", "anzahl"), ("wie viele", "zähl", "anzahl")),
    ("de", ("verschiedene", "unterschiedliche"), ("verschieden", "unterschiedlich", "eindeutig")),
    ("de", ("größer als", "mehr als"), ("größer", "mehr als", "über", "höher", "übersteig")),
    ("de", ("kleiner als", "weniger als"), ("kleiner", "weniger", "unter", "niedriger", "geringer")),
    ("de", ("zwischen",), ("zwischen",)),
    ("de", ("vor dem",), ("vor dem", "vor ", "früher als")),
    ("de", ("am oder nach",), ("am oder nach", "ab dem", "ab ", "seit")),
    ("de", ("höchsten", "meisten"), ("höchst", "meist", "größt", "top")),
    ("de", ("niedrigsten",), ("niedrigst", "kleinst", "geringst", "wenigst")),
    ("de", ("absteigend",), ("absteigend", "höchsten zuerst", "größten zuerst")),
    ("de", ("aufsteigend",), ("aufsteigend", "kleinsten zuerst", "niedrigsten zuerst", "frühesten zuerst")),
    ("de", ("chronologisch",), ("chronologisch", "sortiert", "geordnet", "reihenfolge")),
    ("de", ("keine", "nicht"), ("kein", "nicht", "ohne")),
    ("de", ("runde",), ("rund", "gerundet")),
    ("de", ("monatserster",), ("monatserste", "ersten tag", "monatsanfang", "monatsbeginn")),
    (
        "de",
        ("eine zeile pro",),
        ("zeile", "pro jahr", "pro monat", "je jahr", "je monat", "jedes jahr", "jeden monat"),
    ),
    ("de", (" pro ", " je "), (" pro ", " je ", "jede", "für alle", "nach ")),
]
_NUM = re.compile(r"\d+(?:\.\d+)?")
_QUOTES = re.compile(r"[„“”\"«»]")

SYSTEM = (
    "You rewrite questions about a database so they sound like a real user asked them. "
    "Change the wording and sentence structure, but keep the meaning exactly the same: same filters, same "
    "numbers, same aggregation, same output columns, same sorting and limits. Never add or drop conditions. "
    "Answer with the rewritten question only.\n"
    "Examples:\n"
    "How many customers are there whose market segment is BUILDING? -> "
    "How many customers belong to the BUILDING market segment?\n"
    "For each nation name, compute the average account balance of suppliers. -> "
    "What is the average account balance of suppliers, broken down by nation name?\n"
    "Wie viele Kunden gibt es, deren Marktsegment BUILDING ist? -> "
    "Wie viele Kunden gehören zum Marktsegment BUILDING?\n"
    "Was ist der Durchschnitt von Kontostand über alle Lieferanten? -> "
    "Wie hoch ist der durchschnittliche Kontostand aller Lieferanten?"
)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-zäöüß]+", s.lower())


def detect_lang(s: str) -> str:
    w = _words(s)
    de, en = sum(x in _DE for x in w), sum(x in _EN for x in w)
    return "de" if de > en else "en"


def clean(p: str, template: str) -> str:
    """Drop quote marks the model added around literals (templates never quote values)."""
    p = p.strip()
    if p.lower().startswith(("question:", "frage:")):
        p = p.split(":", 1)[1].strip()
    return p if _QUOTES.search(template) else _QUOTES.sub("", p)


def check_paraphrase(p: str, template: str, lang: str, literals: list[str], terms: list[str]) -> str | None:
    """Return None if acceptable, else the rejection reason."""
    if not p or "\n" in p.strip():
        return "format"
    if _SQL.search(p):
        return "sql"
    if not 0.5 <= len(p) / max(1, len(template)) <= 2.0:
        return "length"
    if p.strip().lower() == template.strip().lower():
        return "unchanged"
    for lit in literals:
        if lit not in p:
            return "literal"
    low = p.lower()
    for t in terms:
        if t.lower() not in low:
            return "term"
    if detect_lang(p) != lang:
        return "language"
    if sorted(_NUM.findall(p)) != sorted(_NUM.findall(template)):
        return "numbers"
    t_low = f" {template.lower()} "
    p_low = f" {low} "
    for clang, triggers, synonyms in CONCEPTS:
        if clang == lang and any(tr in t_low for tr in triggers) and not any(sy in p_low for sy in synonyms):
            return "concept"
    return None


def _prompt(template: str, lang: str, literals: list[str], terms: list[str]) -> str:
    language = "German" if lang == "de" else "English"
    keep = literals + terms
    keep_txt = "; ".join(f'"{k}"' for k in keep) if keep else "(none)"
    return (
        f"Rewrite this question in natural {language}. Use these exact strings unchanged: {keep_txt}.\n"
        f"Question: {template}"
    )


def _key(model: str, template: str, lang: str) -> str:
    return hashlib.sha1(f"{model}\x00{lang}\x00{template}".encode()).hexdigest()


class Paraphraser:
    def __init__(self, model: str = DEFAULT_MODEL, cache: Path = CACHE):
        self.model, self.cache_path = model, cache
        self.cache: dict[str, str] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                d = json.loads(line)
                self.cache[d["key"]] = d["text"]
        self._lock = threading.Lock()
        self.session = requests.Session()

    def generate(self, template: str, lang: str, literals: list[str], terms: list[str], seed: int) -> str:
        k = _key(self.model, template, lang)
        if k in self.cache:
            return self.cache[k]
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _prompt(template, lang, literals, terms)},
            ],
            "options": {"temperature": 0.7, "seed": seed, "num_predict": 160},
        }
        r = self.session.post(OLLAMA_URL, json=body, timeout=120)
        r.raise_for_status()
        text = r.json()["message"]["content"].strip().strip('"').strip()
        with self._lock:
            self.cache[k] = text
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": k, "text": text}, ensure_ascii=False) + "\n")
        return text


def paraphrase_split(df: pl.DataFrame, para: Paraphraser, workers: int) -> tuple[pl.DataFrame, Counter]:
    rows = df.select("question_template", "lang", "literals", "terms", "seed").iter_rows()
    jobs = [(t, lang, json.loads(lit), json.loads(te), seed) for t, lang, lit, te, seed in rows]

    def one(job):
        t, lang, lit, te, seed = job
        try:
            p = clean(para.generate(t, lang, lit, te, seed), t)
        except requests.RequestException:
            return t, "template", "error"
        reason = check_paraphrase(p, t, lang, lit, te)
        return (t, "template", reason) if reason else (p, "paraphrase", "ok")

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, jobs))
    stats = Counter(r for _, _, r in out)
    df = df.with_columns(
        pl.Series("question", [q for q, _, _ in out]),
        pl.Series("question_source", [s for _, s, _ in out]),
    )
    return df, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N rows per split (smoke test)")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    args = ap.parse_args()
    para = Paraphraser(args.model)
    report = {"model": args.model, "splits": {}}
    frames = {}
    for split in args.splits:
        df = pl.read_parquet(DATA_DIR / f"{split}_template.parquet")
        if args.limit:
            df = df.head(args.limit)
        t0 = time.time()
        df, stats = paraphrase_split(df, para, args.workers)
        FinalSchema.validate(df)
        frames[split] = df
        if not args.limit:
            df.write_parquet(DATA_DIR / f"{split}.parquet")
        n = df.height
        report["splits"][split] = {
            "n": n,
            "paraphrase_share": round(stats["ok"] / n, 3),
            "outcomes": dict(stats),
            "seconds": round(time.time() - t0, 1),
        }
        print(split, json.dumps(report["splits"][split]))
    if not args.limit:
        assert_no_leak(frames)
        Path("results/paraphrase.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
