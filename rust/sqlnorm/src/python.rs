use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py(e: crate::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// True iff `sql` is exactly one read-only SELECT/WITH query.
#[pyfunction]
fn is_safe_select(sql: &str) -> bool {
    crate::is_safe_select(sql)
}

/// Canonical SQL text; raises ValueError on parse failure or multiple statements.
#[pyfunction]
fn normalize(sql: &str) -> PyResult<String> {
    crate::normalize(sql).map_err(to_py)
}

/// xxh3-64 of the canonical text, as a 16-char hex string.
#[pyfunction]
fn canonical_hash(sql: &str) -> PyResult<String> {
    crate::canonical_hash(sql)
        .map(|h| format!("{h:016x}"))
        .map_err(to_py)
}

/// Structural features as a dict.
#[pyfunction]
fn features<'py>(py: Python<'py>, sql: &str) -> PyResult<Bound<'py, PyDict>> {
    let f = crate::features(sql).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("num_joins", f.num_joins)?;
    d.set_item("num_tables", f.num_tables)?;
    d.set_item("has_aggregate", f.has_aggregate)?;
    d.set_item("has_group_by", f.has_group_by)?;
    d.set_item("has_order_by", f.has_order_by)?;
    d.set_item("has_limit", f.has_limit)?;
    d.set_item("has_subquery", f.has_subquery)?;
    d.set_item("has_set_op", f.has_set_op)?;
    d.set_item("has_cte", f.has_cte)?;
    Ok(d)
}

/// Batch variant: hashes for many queries in one call (None where parsing fails).
#[pyfunction]
fn canonical_hash_many(sqls: Vec<String>) -> Vec<Option<String>> {
    sqls.iter()
        .map(|s| crate::canonical_hash(s).ok().map(|h| format!("{h:016x}")))
        .collect()
}

#[pymodule]
fn sqlnorm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_safe_select, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_hash_many, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    Ok(())
}
