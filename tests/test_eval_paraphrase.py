import json
import random

from sqlforge.eval.score import Scorer, extract_sql, summarize
from sqlforge.paraphrase.ollama import check_paraphrase, clean, detect_lang
from sqlforge.sampler.generate import Sampler
from sqlforge.sampler.spec import render_sql
from sqlforge.schemas.variant import full_variant, random_variant


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```\nThis returns one.") == "SELECT 1"
    assert extract_sql("Here you go: SELECT a FROM t; -- done") == "SELECT a FROM t"
    assert extract_sql("WITH x AS (SELECT 1) SELECT * FROM x") == "WITH x AS (SELECT 1) SELECT * FROM x"
    assert extract_sql("I cannot answer that.") == ""


def test_scorer_taxonomy(tpch_card):
    s = Scorer()
    v = full_variant(tpch_card).to_json()
    gold = "SELECT n_name FROM nation WHERE n_regionkey = 1"
    assert s.score("SELECT n_name FROM nation WHERE n_regionkey = 1", gold, v, False) == "correct"
    assert s.score("SELECT n_name FROM nation WHERE n_regionkey = 2", gold, v, False) == "wrong_result"
    assert (
        s.score("SELECT COUNT(*) FROM nation", "SELECT SUM(n_regionkey) FROM nation", v, False)
        == "wrong_aggregate"
    )
    assert s.score("SELECT nope FROM nation", gold, v, False) == "wrong_schema_ref"
    assert s.score("SELEC n_name FROM", gold, v, False) == "syntax"
    assert s.score("DROP TABLE nation", gold, v, False) == "not_select"
    assert s.score("", gold, v, False) == "empty"


def test_scorer_uses_variant_names(tpch_card):
    """A prediction is judged in the sample's (renamed) schema: original names must fail there."""
    rng = random.Random(3)
    spec = None
    while spec is None:
        spec = Sampler(tpch_card, rng).sample("count_where")
    v = random_variant(tpch_card, spec.columns(), rng, spec.tables(), p_rename=1.0)
    s = Scorer()
    gold = render_sql(spec, v)
    assert s.score(gold, gold, v.to_json(), False) == "correct"
    assert s.score(render_sql(spec, full_variant(tpch_card)), gold, v.to_json(), False) == "wrong_schema_ref"


def test_summarize_buckets():
    base = {
        "num_joins": 0,
        "has_aggregate": 1,
        "has_group_by": 0,
        "has_subquery": 0,
        "lang": "de",
        "db": "tpch",
        "kind": "agg",
        "question_source": "template",
    }
    rows = [{**base, "outcome": "correct"}, {**base, "outcome": "syntax", "lang": "en"}]
    s = summarize(rows)
    assert s["execution_accuracy"] == 0.5 and s["valid_sql_rate"] == 0.5
    assert s["buckets"]["lang"]["de"]["execution_accuracy"] == 1.0


T = "How many customers are there whose account balance is greater than 5000?"
LIT, TERMS = ["5000"], ["account balance"]


def test_paraphrase_filter_accepts_faithful_rewrite():
    assert (
        check_paraphrase("How many customers have an account balance above 5000?", T, "en", LIT, TERMS)
        is None
    )


def test_paraphrase_filter_rejections():
    assert check_paraphrase("How many customers have a balance above 5000?", T, "en", LIT, TERMS) == "term"
    assert (
        check_paraphrase("How many customers have an account balance above 500?", T, "en", LIT, TERMS)
        == "literal"
    )
    assert (
        check_paraphrase("How many customers have an account balance below 5000?", T, "en", LIT, TERMS)
        == "concept"
    )
    assert (
        check_paraphrase(
            "How many of the 10 customers have an account balance above 5000?", T, "en", LIT, TERMS
        )
        == "numbers"
    )
    assert (
        check_paraphrase("Wie viele Kunden haben einen account balance über 5000?", T, "en", LIT, TERMS)
        == "language"
    )
    assert (
        check_paraphrase("SELECT COUNT(*) FROM customer WHERE account balance > 5000", T, "en", LIT, TERMS)
        == "sql"
    )


def test_clean_and_lang():
    assert (
        clean("Wie viele Kunden haben den Ländername „GERMANY“?", "Wie viele ... GERMANY ist?")
        == "Wie viele Kunden haben den Ländername GERMANY?"
    )
    assert detect_lang("Wie viele Kunden gibt es pro Land?") == "de"
    assert detect_lang("How many customers are there per nation?") == "en"
    assert json.dumps(LIT)
