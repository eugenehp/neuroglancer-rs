//! Tool queries (`ui/tool_query.ts`): which tools a query-defined tool
//! palette lists. A query is clauses, each `+` (include) or `-` (exclude)
//! and a conjunction of `property:value` (or `property:"quoted"`, or
//! `property:/regexp/`, case-insensitive) terms; a bare first clause
//! includes.

use regex::Regex;
use serde_json::Value;

/// `PropertyPredicate`.
#[derive(Debug, Clone)]
pub enum Predicate {
    Equals(String),
    Regexp(Regex),
}

impl PartialEq for Predicate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Predicate::Equals(a), Predicate::Equals(b)) => a == b,
            (Predicate::Regexp(a), Predicate::Regexp(b)) => a.as_str() == b.as_str(),
            _ => false,
        }
    }
}

impl Predicate {
    /// `matchPredicate`.
    pub fn matches(&self, value: &str) -> bool {
        match self {
            Predicate::Equals(s) => s == value,
            Predicate::Regexp(r) => r.is_match(value),
        }
    }
}

/// Offsets into the query text, in characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub begin: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub range: Range,
    pub property: String,
    pub predicate: Predicate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    pub include: bool,
    pub range: Range,
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Query {
    pub clauses: Vec<Clause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub range: Range,
    pub message: String,
}

/// `parsePartialToolQuery`'s result.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialQuery {
    pub raw: String,
    pub query: Query,
    pub errors: Vec<ParseError>,
    pub end_offset: usize,
}

fn token_pattern() -> Regex {
    // `TOKEN_PATTERN`, its lookahead (`(?=$|\s)`) checked by hand.
    Regex::new(r#"^(\s*)(?:(\+|-)|([a-zA-Z]+):(?:("(?:[^"\\]|\\.)*")|([^\s"/][^\s"]*)|/((?:[^/\\]|\\.)*)/))"#).unwrap()
}

/// `parsePartialToolQuery`.
pub fn parse_partial(query: &str) -> PartialQuery {
    let pattern = token_pattern();
    let raw = query.to_owned();
    let mut rest = query;
    let mut offset = 0usize;
    let mut parsed = Query::default();
    let mut current: Option<Clause> = None;
    let mut errors = vec![];
    let chars = |s: &str| s.chars().count();
    let end_clause = |current: &mut Option<Clause>, parsed: &mut Query, errors: &mut Vec<ParseError>, offset: usize| {
        let Some(mut c) = current.take() else { return };
        if c.terms.is_empty() {
            if !c.include {
                errors.push(ParseError { range: Range { begin: c.range.begin, end: offset }, message: "Exclusion clause must have at least one term".into() });
            }
            c.range.end = offset;
        } else {
            c.range.end = c.terms.last().unwrap().range.end;
        }
        parsed.clauses.push(c);
    };
    while let Some(m) = pattern.captures(rest) {
        let whole = m.get(0).unwrap().as_str();
        // `(?=$|\s)` after a term.
        if m.get(3).is_some() {
            let after = &rest[whole.len()..];
            if !(after.is_empty() || after.starts_with(char::is_whitespace)) {
                break;
            }
        }
        let next = offset + chars(whole);
        rest = &rest[whole.len()..];
        let skip = chars(m.get(1).map_or("", |g| g.as_str()));
        if let Some(sign) = m.get(2) {
            end_clause(&mut current, &mut parsed, &mut errors, offset);
            current = Some(Clause { include: sign.as_str() == "+", range: Range { begin: offset + skip, end: 0 }, terms: vec![] });
            offset = next;
            continue;
        }
        let property = m.get(3).unwrap().as_str().to_owned();
        match &current {
            None => current = Some(Clause { include: true, range: Range { begin: offset + skip, end: 0 }, terms: vec![] }),
            Some(c) => {
                if c.terms.iter().any(|t| t.property == property) {
                    errors.push(ParseError {
                        range: Range { begin: offset + skip, end: offset + skip + chars(&property) },
                        message: format!("Property {} cannot be constrained by more than one term in a clause", serde_json::to_string(&property).unwrap()),
                    });
                }
            }
        }
        let predicate = if let Some(q) = m.get(4) {
            Predicate::Equals(serde_json::from_str(q.as_str()).unwrap_or_default())
        } else if let Some(u) = m.get(5) {
            Predicate::Equals(u.as_str().to_owned())
        } else {
            let source = m.get(6).map_or("", |g| g.as_str());
            match Regex::new(&format!("(?i){source}")) {
                Ok(r) => Predicate::Regexp(r),
                Err(e) => {
                    errors.push(ParseError { range: Range { begin: offset + skip + chars(&property) + 2, end: next - 1 }, message: e.to_string() });
                    Predicate::Equals(String::new())
                }
            }
        };
        current.as_mut().unwrap().terms.push(Term { range: Range { begin: offset + skip, end: next }, property, predicate });
        offset = next;
    }
    end_clause(&mut current, &mut parsed, &mut errors, offset);
    offset += chars(&rest[..rest.len() - rest.trim_start().len()]);
    PartialQuery { raw, query: parsed, errors, end_offset: offset }
}

/// `parseToolQuery`: the query, or its errors.
pub fn parse(query: &str) -> Result<Query, Vec<ParseError>> {
    let mut result = parse_partial(query);
    let len = query.chars().count();
    if result.end_offset != len {
        result.errors.push(ParseError { range: Range { begin: result.end_offset, end: len }, message: "Invalid clause/term".into() });
    }
    if result.errors.is_empty() {
        Ok(result.query)
    } else {
        Err(result.errors)
    }
}

/// `matchesTerms`: each term's property a string the predicate accepts.
pub fn matches_terms(tool: &Value, terms: &[Term]) -> bool {
    terms.iter().all(|t| tool.get(&t.property).and_then(Value::as_str).is_some_and(|v| t.predicate.matches(v)))
}

/// `getMatchingTools` over tools listed in binder order (each a palette
/// JSON object, its binder's common properties — `layer`, `layerType` —
/// included): included or excluded clause by clause, keyed by their JSON.
pub fn matching_tools(query: &Query, binders: &[Vec<Value>]) -> Vec<Value> {
    let mut out: Vec<(String, Value)> = vec![];
    for tools in binders {
        for clause in &query.clauses {
            for tool in tools.iter().filter(|t| matches_terms(t, &clause.terms)) {
                let key = serde_json::to_string(tool).unwrap();
                if clause.include {
                    match out.iter_mut().find(|(k, _)| *k == key) {
                        Some(e) => e.1 = tool.clone(),
                        None => out.push((key, tool.clone())),
                    }
                } else {
                    out.retain(|(k, _)| *k != key);
                }
            }
        }
    }
    out.into_iter().map(|(_, v)| v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_and_matches_as_neuroglancer() {
        let q = parse("+").unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert!(q.clauses[0].include && q.clauses[0].terms.is_empty());
        let q = parse("type:shaderControl -layer:/^e/").unwrap();
        assert_eq!(q.clauses.len(), 2);
        assert_eq!(q.clauses[0].terms[0].predicate, Predicate::Equals("shaderControl".into()));
        assert!(q.clauses[1].terms[0].predicate.matches("EM"));
        assert_eq!(parse("type:a type:b").unwrap_err()[0].message, "Property \"type\" cannot be constrained by more than one term in a clause");
        assert_eq!(parse("-").unwrap_err()[0].message, "Exclusion clause must have at least one term");
        assert_eq!(parse("type:a x").unwrap_err()[0].range, Range { begin: 7, end: 8 });
        assert!(parse("layer:\"a b\"").is_ok());
        let tools = vec![
            vec![json!({"type": "dimension", "dimension": "x"})],
            vec![json!({"layer": "em", "type": "opacity", "layerType": "image"}), json!({"layer": "em", "type": "shaderControl", "control": "normalized", "layerType": "image"})],
        ];
        let all = matching_tools(&parse("+").unwrap(), &tools);
        assert_eq!(all.len(), 3);
        let shader = matching_tools(&parse("type:shaderControl").unwrap(), &tools);
        assert_eq!(shader, vec![tools[1][1].clone()]);
        let some = matching_tools(&parse("+ -type:dimension").unwrap(), &tools);
        assert_eq!(some.len(), 2);
    }
}
