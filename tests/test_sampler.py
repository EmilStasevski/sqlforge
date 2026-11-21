import random

import duckdb
import pytest
import sqlnorm

from sqlforge.nlgen.render import render_question
from sqlforge.sampler.generate import KIND_WEIGHTS, Sampler
from sqlforge.sampler.spec import Filter, Sel, Spec, render_sql
from sqlforge.schemas.variant import Variant, full_variant, random_variant
from sqlforge.verify.execute import QueryTimeout, Verifier, run_query


def test_card_loads_types_and_joins(tpch_card):
    assert tpch_card.col("customer", "c_acctbal").role == "num"
    assert tpch_card.col("customer", "c_acctbal").type.startswith("DECIMAL")
    paths = tpch_card.m1_paths("customer")
    assert any([e[1].table for e in p] == ["nation", "region"] for p in paths)


def test_every_kind_renders_safe_sql_and_question(tpch_card):
    rng = random.Random(0)
    s = Sampler(tpch_card, rng)
    fv = full_variant(tpch_card)
    for kind in KIND_WEIGHTS:
        for _ in range(20):
            spec = s.sample(kind)
            if spec is not None:
                break
        assert spec is not None, kind
        sql = render_sql(spec, fv)
        assert sqlnorm.is_safe_select(sql), sql
        for lang in ("en", "de"):
            r = render_question(spec, tpch_card, lang, random.Random(1))
            for lit in r.literals:
                assert lit in r.question, (lit, r.question)
            for t in r.terms:
                assert t in r.question


def test_sampling_is_deterministic(tpch_card):
    def run():
        s = Sampler(tpch_card, random.Random(42))
        return [render_sql(sp, full_variant(tpch_card)) for sp in (s.sample() for _ in range(30)) if sp]

    assert run() == run()


def test_renamed_variant_matches_original(tpch_card):
    ver = Verifier(tpch_card)
    rng = random.Random(7)
    s = Sampler(tpch_card, rng)
    checked = 0
    for _ in range(200):
        spec = s.sample()
        if spec is None:
            continue
        v = random_variant(tpch_card, spec.columns(), rng, spec.tables(), p_rename=1.0)
        assert not v.is_identity
        verdict = ver.check(spec, v)
        assert verdict.reason != "variant", render_sql(spec, v)
        checked += verdict.ok
    assert checked > 50


def test_variant_hides_unlisted_columns(tpch_card):
    v = Variant("tpch", [], [])
    v = Variant.from_json(full_variant(tpch_card).to_json())
    v.tables = [t for t in v.tables if t.orig == "nation"]
    v.tables[0].columns = [c for c in v.tables[0].columns if c[0] != "n_comment"]
    con = v.connect(str(tpch_card.db_path))
    assert con.execute("SELECT COUNT(*) FROM nation").fetchone()[0] == 25
    with pytest.raises(duckdb.BinderException):
        con.execute("SELECT n_comment FROM nation")
    with pytest.raises(duckdb.CatalogException):
        con.execute("SELECT * FROM customer")


def test_verifier_rejects_ties_and_empty(tpch_card):
    ver, fv = Verifier(tpch_card), full_variant(tpch_card)
    n = tpch_card.tables["nation"]
    ties = Spec(
        "topk",
        "nation",
        select=[Sel(None, n.col("n_name")), Sel(None, n.col("n_regionkey"))],
        order_by=(1, "DESC"),
        limit=2,
    )
    assert ver.check(ties, fv).reason == "ties"
    empty = Spec(
        "count_where",
        "nation",
        select=[Sel(None, n.col("n_name"))],
        filters=[Filter(n.col("n_name"), "=", ("ATLANTIS",))],
    )
    assert ver.check(empty, fv).reason == "rows"
    ok = Spec("count_all", "nation", select=[Sel("COUNT")])
    assert ver.check(ok, fv).ok


def test_run_query_timeout(tpch_card):
    con = duckdb.connect()
    with pytest.raises(QueryTimeout):
        run_query(con, "SELECT COUNT(*) FROM range(100000000) a, range(100000) b", timeout_s=0.3)
