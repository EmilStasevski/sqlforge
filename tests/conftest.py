import duckdb
import pytest

from sqlforge.schemas import card as card_mod


@pytest.fixture(scope="session")
def tpch_dir(tmp_path_factory):
    """Tiny TPC-H (sf=0.01) so sampler/verifier tests run without the real data build."""
    d = tmp_path_factory.mktemp("db")
    con = duckdb.connect(str(d / "tpch.duckdb"))
    con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01);")
    con.close()
    return d


@pytest.fixture
def tpch_card(tpch_dir, monkeypatch):
    monkeypatch.setattr(card_mod, "DB_DIR", tpch_dir)
    return card_mod.load_card("tpch")
