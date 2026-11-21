use super::*;

#[test]
fn safe_select_accepts_queries() {
    assert!(is_safe_select("SELECT 1"));
    assert!(is_safe_select(
        "select a from t where b > 3 order by a limit 5;"
    ));
    assert!(is_safe_select("WITH x AS (SELECT 1 AS a) SELECT a FROM x"));
    assert!(is_safe_select("SELECT a FROM t UNION ALL SELECT a FROM u"));
}

#[test]
fn safe_select_rejects_everything_else() {
    for sql in [
        "DROP TABLE t",
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "CREATE TABLE t (a INT)",
        "SELECT 1; SELECT 2",
        "SELECT 1; DROP TABLE t",
        "SELECT * INTO t2 FROM t",
        "COPY t TO 'x.csv'",
        "ATTACH 'x.db'",
        "not sql at all",
        "",
    ] {
        assert!(!is_safe_select(sql), "should reject: {sql}");
    }
}

#[test]
fn normalize_is_case_and_whitespace_insensitive() {
    let a = normalize("select  C_NAME from Customer\n where c_acctbal>100").unwrap();
    let b = normalize("SELECT c_name FROM customer WHERE c_acctbal > 100").unwrap();
    assert_eq!(a, b);
    assert_eq!(
        canonical_hash("SELECT 1").unwrap(),
        canonical_hash("select   1").unwrap()
    );
}

#[test]
fn normalize_keeps_literals_and_quoted_identifiers() {
    let a = normalize("SELECT * FROM city WHERE name = 'Köln'").unwrap();
    assert!(a.contains("'Köln'"), "{a}");
    assert_ne!(
        canonical_hash("SELECT * FROM t WHERE x = 'A'").unwrap(),
        canonical_hash("SELECT * FROM t WHERE x = 'a'").unwrap()
    );
    let q = normalize("SELECT \"MixedCase\" FROM t").unwrap();
    assert!(q.contains("\"MixedCase\""), "{q}");
}

#[test]
fn normalize_ignores_comments() {
    assert_eq!(
        normalize("SELECT a -- comment\nFROM t").unwrap(),
        normalize("SELECT a FROM t").unwrap()
    );
}

#[test]
fn features_basic() {
    let f = features(
        "SELECT n.n_name, AVG(c.c_acctbal) FROM customer c JOIN nation n ON c.c_nationkey = n.n_nationkey \
         GROUP BY n.n_name ORDER BY 2 DESC LIMIT 3",
    )
    .unwrap();
    assert_eq!(f.num_joins, 1);
    assert_eq!(f.num_tables, 2);
    assert!(f.has_aggregate && f.has_group_by && f.has_order_by && f.has_limit);
    assert!(!f.has_subquery && !f.has_set_op && !f.has_cte);
}

#[test]
fn features_subquery_and_plain() {
    let f = features(
        "SELECT c_name FROM customer WHERE c_acctbal > (SELECT AVG(c_acctbal) FROM customer)",
    )
    .unwrap();
    assert!(f.has_subquery && f.has_aggregate && !f.has_group_by);
    assert_eq!(f.num_tables, 1);

    let f = features("SELECT * FROM t").unwrap();
    assert_eq!(
        f,
        Features {
            num_tables: 1,
            ..Default::default()
        }
    );
}

#[test]
fn window_functions_are_not_aggregates() {
    let f = features("SELECT a, SUM(b) OVER (PARTITION BY a) FROM t").unwrap();
    assert!(!f.has_aggregate);
}

#[test]
fn comma_join_and_derived_table() {
    let f = features("SELECT * FROM a, b WHERE a.id = b.id").unwrap();
    assert_eq!(f.num_joins, 1);
    let f = features("SELECT x FROM (SELECT 1 AS x) s").unwrap();
    assert!(f.has_subquery);
}

#[test]
fn errors_are_reported() {
    assert!(matches!(
        normalize("SELECT FROM WHERE"),
        Err(Error::Parse(_))
    ));
    assert_eq!(
        normalize("SELECT 1; SELECT 2"),
        Err(Error::NotSingleStatement(2))
    );
}
