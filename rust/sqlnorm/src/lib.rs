//! sqlnorm: parse / guard / normalize / hash / feature-extract DuckDB SQL.
//!
//! Pure-Rust core (tested with `cargo test`); PyO3 bindings behind the `python` feature.

use std::collections::BTreeSet;
use std::ops::ControlFlow;

use sqlparser::ast::{
    Expr, FunctionArguments, Query, SetExpr, Statement, TableFactor, TableWithJoins, Visit, Visitor,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, Tokenizer};

const AGGREGATES: &[&str] = &[
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "median",
    "stddev",
    "stddev_samp",
    "stddev_pop",
    "variance",
    "var_samp",
    "var_pop",
    "string_agg",
    "list",
    "array_agg",
    "mode",
    "quantile_cont",
    "quantile_disc",
    "count_if",
    "any_value",
    "first",
    "last",
    "arg_max",
    "arg_min",
    "bool_and",
    "bool_or",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Parse(String),
    NotSingleStatement(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "parse error: {e}"),
            Error::NotSingleStatement(n) => write!(f, "expected exactly 1 statement, got {n}"),
        }
    }
}

impl std::error::Error for Error {}

fn parse_one(sql: &str) -> Result<Statement, Error> {
    let mut stmts =
        Parser::parse_sql(&DuckDbDialect {}, sql).map_err(|e| Error::Parse(e.to_string()))?;
    if stmts.len() != 1 {
        return Err(Error::NotSingleStatement(stmts.len()));
    }
    Ok(stmts.remove(0))
}

/// True iff `sql` is exactly one read-only query (SELECT / WITH ... SELECT / set operations).
/// Rejects DDL, DML, multiple statements, `SELECT ... INTO`, and any nested non-query statement.
pub fn is_safe_select(sql: &str) -> bool {
    let Ok(stmt) = parse_one(sql) else {
        return false;
    };
    let Statement::Query(q) = &stmt else {
        return false;
    };
    query_is_read_only(q)
}

fn query_is_read_only(q: &Query) -> bool {
    if let Some(with) = &q.with {
        if !with
            .cte_tables
            .iter()
            .all(|cte| query_is_read_only(&cte.query))
        {
            return false;
        }
    }
    setexpr_is_read_only(&q.body)
}

fn setexpr_is_read_only(s: &SetExpr) -> bool {
    match s {
        SetExpr::Select(sel) => sel.into.is_none(),
        SetExpr::Query(q) => query_is_read_only(q),
        SetExpr::SetOperation { left, right, .. } => {
            setexpr_is_read_only(left) && setexpr_is_read_only(right)
        }
        SetExpr::Values(_) => true,
        SetExpr::Table(_) => true,
        // Insert / Update / Delete / Merge bodies and anything new are rejected.
        _ => false,
    }
}

/// Canonical text: unquoted identifiers lower-cased, keywords upper-cased by the AST printer,
/// whitespace and comments collapsed. Literals are preserved (they change semantics).
pub fn normalize(sql: &str) -> Result<String, Error> {
    let dialect = DuckDbDialect {};
    let tokens = Tokenizer::new(&dialect, sql)
        .tokenize()
        .map_err(|e| Error::Parse(e.to_string()))?;
    let mut lowered = String::with_capacity(sql.len());
    for tok in tokens {
        match tok {
            Token::Word(mut w) if w.quote_style.is_none() => {
                w.value = w.value.to_lowercase();
                lowered.push_str(&Token::Word(w).to_string());
            }
            Token::Whitespace(_) => lowered.push(' '),
            other => lowered.push_str(&other.to_string()),
        }
    }
    let stmt = parse_one(&lowered)?;
    Ok(stmt.to_string())
}

/// 64-bit xxh3 of the canonical text (dedup + leak check key).
pub fn canonical_hash(sql: &str) -> Result<u64, Error> {
    Ok(xxhash_rust::xxh3::xxh3_64(normalize(sql)?.as_bytes()))
}

/// Structural features used for evaluation buckets.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Features {
    pub num_joins: u32,
    pub num_tables: u32,
    pub has_aggregate: bool,
    pub has_group_by: bool,
    pub has_order_by: bool,
    pub has_limit: bool,
    pub has_subquery: bool,
    pub has_set_op: bool,
    pub has_cte: bool,
}

#[derive(Default)]
struct FeatureVisitor {
    f: Features,
    tables: BTreeSet<String>,
    query_depth: u32,
}

impl Visitor for FeatureVisitor {
    type Break = ();

    fn pre_visit_query(&mut self, q: &Query) -> ControlFlow<()> {
        self.query_depth += 1;
        if self.query_depth > 1 {
            self.f.has_subquery = true;
        }
        if q.with.is_some() {
            self.f.has_cte = true;
        }
        if q.order_by.is_some() {
            self.f.has_order_by = true;
        }
        if q.limit_clause.is_some() || q.fetch.is_some() {
            self.f.has_limit = true;
        }
        count_setexpr(&q.body, &mut self.f);
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _q: &Query) -> ControlFlow<()> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, rel: &sqlparser::ast::ObjectName) -> ControlFlow<()> {
        self.tables.insert(rel.to_string().to_lowercase());
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, e: &Expr) -> ControlFlow<()> {
        if let Expr::Function(func) = e {
            let name = func.name.to_string().to_lowercase();
            let is_window = func.over.is_some();
            if !is_window && AGGREGATES.contains(&name.as_str()) {
                self.f.has_aggregate = true;
            }
            if let FunctionArguments::Subquery(_) = func.args {
                self.f.has_subquery = true;
            }
        }
        ControlFlow::Continue(())
    }
}

fn count_setexpr(s: &SetExpr, f: &mut Features) {
    match s {
        SetExpr::Select(sel) => {
            f.num_joins += sel
                .from
                .iter()
                .map(|t: &TableWithJoins| t.joins.len() as u32)
                .sum::<u32>();
            // Comma joins: FROM a, b counts as one join.
            f.num_joins += sel.from.len().saturating_sub(1) as u32;
            for twj in &sel.from {
                if matches!(twj.relation, TableFactor::Derived { .. }) {
                    f.has_subquery = true;
                }
            }
            if !matches!(&sel.group_by, sqlparser::ast::GroupByExpr::Expressions(v, _) if v.is_empty())
            {
                f.has_group_by = true;
            }
        }
        SetExpr::SetOperation { left, right, .. } => {
            f.has_set_op = true;
            count_setexpr(left, f);
            count_setexpr(right, f);
        }
        _ => {}
    }
}

pub fn features(sql: &str) -> Result<Features, Error> {
    let stmt = parse_one(sql)?;
    let mut v = FeatureVisitor::default();
    let _ = stmt.visit(&mut v);
    v.f.num_tables = v.tables.len() as u32;
    Ok(v.f)
}

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests;
