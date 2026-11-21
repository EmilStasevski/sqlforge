import pytest
import sqlnorm


def test_guard():
    assert sqlnorm.is_safe_select("SELECT n_name FROM nation")
    assert not sqlnorm.is_safe_select("SELECT 1; DROP TABLE nation")
    assert not sqlnorm.is_safe_select("DELETE FROM nation")


def test_hash_stable_under_formatting():
    a = sqlnorm.canonical_hash("select N_NAME from nation  where n_regionkey=1")
    b = sqlnorm.canonical_hash("SELECT n_name\nFROM nation WHERE n_regionkey = 1")
    assert a == b and len(a) == 16


def test_hash_many_marks_failures():
    out = sqlnorm.canonical_hash_many(["SELECT 1", "SELEC nope", "select 1"])
    assert out[0] == out[2] and out[1] is None


def test_errors_raise_valueerror():
    with pytest.raises(ValueError):
        sqlnorm.normalize("SELECT FROM WHERE")


def test_features_dict():
    f = sqlnorm.features("SELECT COUNT(*) FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey")
    assert f["num_joins"] == 1 and f["num_tables"] == 2 and f["has_aggregate"]
