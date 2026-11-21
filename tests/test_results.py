from decimal import Decimal

from sqlforge.verify.results import results_match


def test_order_insensitive_unless_ordered():
    a, b = [(1, "x"), (2, "y")], [(2, "y"), (1, "x")]
    assert results_match(a, b, ordered=False)
    assert not results_match(a, b, ordered=True)


def test_column_permutation_and_numeric_tolerance():
    assert results_match([("DE", Decimal("10.50"))], [(10.5000000001, "DE")], ordered=False)
    assert not results_match([("DE", 10.5)], [("DE", 10.6)], ordered=False)


def test_duplicates_and_shapes_matter():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)
    assert not results_match([(1, 2)], [(1,)], ordered=False)
    assert results_match([], [], ordered=False)
