//! Viewer state in URLs: `urlSafeToJSON` (`src/util/json.ts`) and the
//! `#!…` fragment forms of `UrlHashBinding.updateFromUrlHash`
//! (`src/ui/url_hash_binding.ts`).
//!
//! Share links write `_` for `,` and `'` for `"` to survive URL escaping;
//! outside string literals `&`, `_` and `,` all read as `,`, and single-quoted
//! strings become double-quoted.

use anyhow::{anyhow, bail, Result};

/// A quoted string literal starting at `chars[start]` (which is `q`), as
/// `('(?:[^'\\]|(?:\\.))*')` matches it: its end (exclusive), if terminated.
/// `.` does not match line terminators.
fn literal_end(chars: &[char], start: usize, q: char) -> Option<usize> {
    let mut i = start + 1;
    while i < chars.len() {
        match chars[i] {
            c if c == q => return Some(i + 1),
            '\\' => {
                let next = *chars.get(i + 1)?;
                if matches!(next, '\n' | '\r' | '\u{2028}' | '\u{2029}') {
                    return None;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    None
}

/// `convertStringLiteral(x, "'", '"', DOUBLE_QUOTE_PATTERN)`: a single-quoted
/// literal to a double-quoted one.
fn single_to_double(lit: &[char]) -> String {
    let inner = &lit[1..lit.len() - 1];
    let mut s = String::from('"');
    let mut i = 0;
    // `^((?:[^"'\\]|(?:\\[^']))*)("|\\')`: copy until a bare `"` (escape it)
    // or `\'` (unescape it); a match failing leaves the rest as is.
    loop {
        let mut j = i;
        let found = loop {
            match inner.get(j) {
                None => break None,
                Some('"') => break Some((j, 1, true)),
                Some('\\') if inner.get(j + 1) == Some(&'\'') => break Some((j, 2, false)),
                Some('\\') if inner.get(j + 1).is_some() => j += 2,
                Some('\\') => break None,
                Some('\'') => break None,
                Some(_) => j += 1,
            }
        };
        match found {
            None => {
                s.extend(&inner[i..]);
                break;
            }
            Some((at, len, bare_double)) => {
                s.extend(&inner[i..at]);
                if bare_double {
                    s.push_str("\\\"");
                } else {
                    s.push('\'');
                }
                i = at + len;
            }
        }
    }
    s.push('"');
    s
}

/// `urlSafeToJSON`.
pub fn url_safe_to_json(x: &str) -> String {
    let chars: Vec<char> = x.chars().collect();
    let mut out = String::with_capacity(x.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' || c == '"' {
            if let Some(end) = literal_end(&chars, i, c) {
                if c == '\'' {
                    out.push_str(&single_to_double(&chars[i..end]));
                } else {
                    out.extend(&chars[i..end]);
                }
                i = end;
                continue;
            }
        }
        out.push(if matches!(c, '&' | '_' | ',') { ',' } else { c });
        i += 1;
    }
    out
}

/// `urlSafeParse`.
pub fn url_safe_parse(x: &str) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&url_safe_to_json(x))?)
}

/// `decodeURIComponent`: `%XX` sequences as UTF-8; malformed ones are an
/// error (`URIError`).
pub fn decode_uri_component(s: &str) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3).ok_or_else(|| anyhow!("URI malformed"))?;
            out.push(u8::from_str_radix(hex, 16).map_err(|_| anyhow!("URI malformed"))?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| anyhow!("URI malformed"))
}

/// What a URL fragment holds.
#[derive(Debug, Clone, PartialEq)]
pub enum Fragment {
    /// `#!{…}`: the whole state.
    State(serde_json::Value),
    /// `#!+{…}`: changes over the current state.
    Update(serde_json::Value),
    /// `#!scheme://…`: a URL to fetch the state from.
    Remote(String),
    /// Nothing (`""`, `#`, `#!`): the default state.
    Empty,
}

/// `updateFromUrlHash`'s reading of `fragment` (from `#`, or empty).
pub fn parse_fragment(fragment: &str) -> Result<Fragment> {
    let s = fragment;
    if s.is_empty() || s == "#" || s == "#!" {
        return Ok(Fragment::Empty);
    }
    if let Some(rest) = s.strip_prefix("#!") {
        // `^#!([a-z][a-z\d+-.]*):\/\/`
        let scheme_len = rest
            .char_indices()
            .take_while(|&(i, c)| if i == 0 { c.is_ascii_lowercase() } else { c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '+' | '-' | '.') })
            .count();
        if scheme_len > 0 && rest[scheme_len..].starts_with("://") {
            return Ok(Fragment::Remote(rest.to_owned()));
        }
        let (update, body) = match rest.strip_prefix('+') {
            Some(b) => (true, b),
            None => (false, rest),
        };
        let value = url_safe_parse(&decode_uri_component(body)?)?;
        if !value.is_object() {
            bail!("Expected object, but received: {value}");
        }
        return Ok(if update { Fragment::Update(value) } else { Fragment::State(value) });
    }
    bail!("URL hash is expected to be of the form \"#!{{...}}\" or \"#!+{{...}}\".")
}

/// `JSON.stringify`: JavaScript's number formatting and string escapes.
pub fn js_stringify(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(_) => crate::js::to_string(v),
        Value::String(s) => {
            let mut out = String::from('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{8}' => out.push_str("\\b"),
                    '\u{c}' => out.push_str("\\f"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Value::Array(a) => format!("[{}]", a.iter().map(js_stringify).collect::<Vec<_>>().join(",")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter().map(|(k, v)| format!("{}:{}", js_stringify(&Value::String(k.clone())), js_stringify(v))).collect::<Vec<_>>().join(",")
        ),
    }
}

/// `encodeURI`, then `!'()*;,` percent-encoded too (`encodeFragment`).
pub fn encode_fragment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let keep = c.is_ascii_alphanumeric() || "-_.~/?:@&=+$#".contains(c);
        if keep {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

/// `encodeStateAsFragment`: the text after `#!` in a share link.
pub fn encode_state_as_fragment(state: &serde_json::Value) -> String {
    encode_fragment(&js_stringify(state))
}
