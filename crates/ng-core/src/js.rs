//! JavaScript numeric semantics, where neuroglancer's behavior depends on them.

/// `Math.round`: halves round toward +∞ (`-2.5` → `-2`), unlike `f64::round`,
/// which rounds them away from zero.
pub fn round(x: f64) -> f64 {
    let f = x.floor();
    // Exact: `x - floor(x)` loses nothing for finite `x`.
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// `ToUint32` (`x >>> 0`): truncate toward zero, then wrap modulo 2^32. NaN
/// and infinities become 0.
pub fn to_uint32(x: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    x.trunc().rem_euclid(4_294_967_296.0) as u32
}

/// `Object.is`: like `==`, except NaN equals NaN and `0` differs from `-0`.
pub fn object_is(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    a == b && a.is_sign_negative() == b.is_sign_negative()
}

/// `Math.fround`: what storing into a `Float32Array` does.
pub fn fround(x: f64) -> f64 {
    x as f32 as f64
}

/// A JSON number as `JSON.stringify` would write it: integral values without
/// a fractional part (`4`, not `4.0`). Non-finite values become `null`.
pub fn number(x: f64) -> serde_json::Value {
    const MAX_SAFE: f64 = 9_007_199_254_740_991.0;
    if x.fract() == 0.0 && x.abs() <= MAX_SAFE {
        serde_json::Value::from(x as i64)
    } else {
        serde_json::Number::from_f64(x).map_or(serde_json::Value::Null, serde_json::Value::Number)
    }
}

/// `Math.sin`, `Math.cos`, `Math.tan`, `Math.log2`, `Math.exp`: V8 implements these with
/// an fdlibm port, as the `libm` crate (from musl) does; the platform's libm
/// can differ in the last place.
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `parseFloat(string)`: skip leading whitespace, then read the longest prefix
/// that is a decimal literal or `Infinity` (with optional sign); NaN if there
/// is none. `"12px"` is 12, `" -1e3x"` is -1000, `"0x10"` is 0.
pub fn parse_float(s: &str) -> f64 {
    let s = s.trim_start_matches(is_js_whitespace);
    let b = s.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+' | b'-')) {
        i = 1;
    }
    if s[i..].starts_with("Infinity") {
        return if b.first() == Some(&b'-') { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    let digits = |mut j: usize| {
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let int_end = digits(i);
    let mut end = int_end;
    let mut has_digits = int_end > i;
    if end < b.len() && b[end] == b'.' {
        let frac_end = digits(end + 1);
        if frac_end > end + 1 || has_digits {
            has_digits |= frac_end > end + 1;
            end = frac_end;
        }
    }
    if !has_digits {
        return f64::NAN;
    }
    if end < b.len() && (b[end] == b'e' || b[end] == b'E') {
        let mut j = end + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_end = digits(j);
        if exp_end > j {
            end = exp_end;
        }
    }
    // Rust's parser accepts exactly this grammar, apart from a trailing `.`.
    s[..end].trim_end_matches('.').parse().unwrap_or(f64::NAN)
}

/// `parseInt(string)` (no radix): skip leading whitespace, an optional sign,
/// then `0x`/`0X` and hex digits, or decimal digits; the longest such prefix.
/// NaN if there are no digits. `"12.7"` is 12, `"1e3"` is 1, `"0x1f"` is 31.
pub fn parse_int(s: &str) -> f64 {
    parse_int_with(s, true)
}

/// `parseInt(string, 10)`: decimal digits only, so `"0x1f"` is 0.
pub fn parse_int10(s: &str) -> f64 {
    parse_int_with(s, false)
}

fn parse_int_with(s: &str, hex: bool) -> f64 {
    let s = s.trim_start_matches(is_js_whitespace);
    let (negative, s) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let (radix, s) = match s.get(..2) {
        Some("0x" | "0X") if hex => (16, &s[2..]),
        _ => (10, s),
    };
    let digits: String = s.chars().take_while(|c| c.is_digit(radix)).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    // Accumulated in doubles, as the spec's mathematical value rounds.
    let value = if radix == 10 {
        digits.parse::<f64>().unwrap_or(f64::NAN)
    } else {
        digits.chars().fold(0.0, |acc, c| acc * 16.0 + c.to_digit(16).unwrap() as f64)
    };
    if negative { -value } else { value }
}

/// ECMAScript `WhiteSpace` and `LineTerminator`.
fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}'
            | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'
    )
}

/// `Number.prototype.toString()` (ECMAScript `Number::toString`, radix 10):
/// the shortest digits that round-trip, positioned as the spec says —
/// `0.0000001` is `1e-7`, `1e21` is `1e+21`, `123456789012345683968` is
/// `123456789012345680000`.
pub fn number_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sign = if x < 0.0 { "-" } else { "" };
    // Rust's `{:e}` gives the shortest round-trip digits: `d.ddde±x`.
    let e = format!("{:e}", x.abs());
    let (mantissa, exp) = e.split_once('e').unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp.parse::<i32>().unwrap() + 1;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let exp = if e >= 0 { format!("+{e}") } else { format!("{e}") };
        if k == 1 {
            format!("{digits}e{exp}")
        } else {
            format!("{}.{}e{exp}", &digits[..1], &digits[1..])
        }
    };
    format!("{sign}{body}")
}

/// `|x|`'s exact decimal digits, without the point, and the power of ten of
/// the first: `x = 0.d₀d₁d₂… × 10^(e+1)`. A double's expansion ends within
/// 767 significant digits, so 800 hold all of it.
fn exact_digits(x: f64) -> (Vec<u8>, i32) {
    let s = format!("{:.800e}", x.abs());
    let (m, e) = s.split_once('e').unwrap();
    (m.bytes().filter(u8::is_ascii_digit).map(|b| b - b'0').collect(), e.parse().unwrap())
}

/// The first `n` of `digits` rounded half up (ECMAScript: "if there are two
/// such n, pick the larger"); a carry out of the first digit prepends a 1,
/// reported as `true`.
fn round_digits(digits: &[u8], n: usize) -> (Vec<u8>, bool) {
    let mut out: Vec<u8> = digits.iter().copied().chain(std::iter::repeat(0)).take(n).collect();
    if digits.get(n).is_some_and(|&d| d >= 5) {
        let mut i = n;
        loop {
            if i == 0 {
                out.insert(0, 1);
                out.truncate(n.max(1));
                return (out, true);
            }
            i -= 1;
            if out[i] == 9 {
                out[i] = 0;
            } else {
                out[i] += 1;
                break;
            }
        }
    }
    (out, false)
}

fn digit_string(d: &[u8]) -> String {
    d.iter().map(|d| (b'0' + d) as char).collect()
}

/// `Number.prototype.toFixed(f)`.
pub fn to_fixed(x: f64, f: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return number_to_string(x);
    }
    let sign = if x < 0.0 { "-" } else { "" };
    // Digits up to 10^-f: the integer part's (e + 1) plus f.
    let (digits, e) = if x == 0.0 { (vec![0], 0) } else { exact_digits(x) };
    let int_len = e + 1;
    let keep = int_len + f as i32;
    let m = if keep < 0 {
        "0".to_owned()
    } else {
        let (r, carry) = round_digits(&digits, keep as usize);
        let mut s = digit_string(&r);
        if carry && keep > 0 {
            s.push('0');
        }
        s
    };
    // `m` is the integer n with x ≈ n / 10^f.
    let m = format!("{:0>width$}", m.trim_start_matches('0'), width = f + 1);
    let body = if f == 0 { m } else { format!("{}.{}", &m[..m.len() - f], &m[m.len() - f..]) };
    format!("{sign}{body}")
}

/// `Number.prototype.toPrecision(p)`.
pub fn to_precision(x: f64, p: usize) -> String {
    if !x.is_finite() {
        return number_to_string(x);
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let (digits, mut e) = if x == 0.0 { (vec![0; p], 0) } else { exact_digits(x) };
    let (m, carry) = round_digits(&digits, p);
    if carry {
        e += 1;
    }
    let m = digit_string(&m);
    let body = if e < -6 || e >= p as i32 {
        let exp = if e >= 0 { format!("+{e}") } else { format!("{e}") };
        if p == 1 {
            format!("{m}e{exp}")
        } else {
            format!("{}.{}e{exp}", &m[..1], &m[1..])
        }
    } else if e == p as i32 - 1 {
        m
    } else if e >= 0 {
        format!("{}.{}", &m[..e as usize + 1], &m[e as usize + 1..])
    } else {
        format!("0.{}{m}", "0".repeat((-(e + 1)) as usize))
    };
    format!("{sign}{body}")
}

/// `String(x)` for a JSON value, as `parseFloat` sees an argument.
pub fn to_string(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => number_to_string(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .map(|e| if e.is_null() { String::new() } else { to_string(e) })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".into(),
    }
}

/// `verifyFloat` (`src/util/json.ts`): a number, or a string `parseFloat`
/// reads a number from.
pub fn verify_float(v: &serde_json::Value) -> anyhow::Result<f64> {
    let x = match v {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        serde_json::Value::String(s) => parse_float(s),
        _ => f64::NAN,
    };
    anyhow::ensure!(!x.is_nan(), "Expected floating-point number, but received: {v}.");
    Ok(x)
}

/// `verifyFiniteFloat`.
pub fn verify_finite_float(v: &serde_json::Value) -> anyhow::Result<f64> {
    let x = verify_float(v)?;
    anyhow::ensure!(x.is_finite(), "Expected finite floating-point number, but received: {x}.");
    Ok(x)
}

/// `verifyFinitePositiveFloat`.
pub fn verify_finite_positive_float(v: &serde_json::Value) -> anyhow::Result<f64> {
    let x = verify_float(v)?;
    anyhow::ensure!(
        x.is_finite() && x > 0.0,
        "Expected positive finite floating-point number, but received: {x}."
    );
    Ok(x)
}

#[cfg(test)]
mod fixed_tests {
    use super::*;

    #[test]
    fn to_fixed_and_to_precision_match_v8() {
        // Values from Node: `(x).toFixed(f)` / `(x).toPrecision(p)`.
        assert_eq!(to_fixed(4.0, 6), "4.000000");
        assert_eq!(to_fixed(1.005, 2), "1.00");
        assert_eq!(to_fixed(2.5, 0), "3");
        assert_eq!(to_fixed(-2.5, 0), "-3");
        assert_eq!(to_fixed(0.000001, 6), "0.000001");
        assert_eq!(to_fixed(999.9999995, 6), "999.999999");
        assert_eq!(to_fixed(9.9999999, 6), "10.000000");
        assert_eq!(to_fixed(0.0, 2), "0.00");
        assert_eq!(to_fixed(-0.001, 2), "-0.00");
        assert_eq!(to_precision(1234565.0, 6), "1.23457e+6");
        assert_eq!(to_precision(0.5, 6), "0.500000");
        assert_eq!(to_precision(1500.0, 6), "1500.00");
        assert_eq!(to_precision(0.0000012345, 6), "0.00000123450");
        assert_eq!(to_precision(1.2345e-7, 6), "1.23450e-7");
        assert_eq!(to_precision(999999.5, 6), "1.00000e+6");
        assert_eq!(to_precision(123456.0, 6), "123456");
    }
}

#[cfg(test)]
mod number_to_string_tests {
    use super::number_to_string as f;

    #[test]
    fn formats_as_javascript() {
        for (x, s) in [
            (1.5, "1.5"),
            (-0.0, "0"),
            (1e21, "1e+21"),
            (1e-7, "1e-7"),
            (1e-6, "0.000001"),
            (4e-9, "4e-9"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456789012345683968.0, "123456789012345680000"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (-1.5e-10, "-1.5e-10"),
            (100.0, "100"),
            (1e20, "100000000000000000000"),
            (9007199254740992.0, "9007199254740992"),
            (123.456, "123.456"),
        ] {
            assert_eq!(f(x), s, "{x:e}");
        }
    }
}

/// `verifyInt`: `parseInt(String(v), 10)`, so `2.5` is 2 and `"7"` is 7.
pub fn verify_int(v: &serde_json::Value) -> anyhow::Result<f64> {
    let x = parse_int10(&to_string(v));
    anyhow::ensure!(x.is_finite(), "Expected integer, but received: {v}.");
    Ok(x)
}

/// `verifyPositiveInt`.
pub fn verify_positive_int(v: &serde_json::Value) -> anyhow::Result<u64> {
    let x = verify_int(v)?;
    anyhow::ensure!(x > 0.0, "Expected positive integer, but received: {}.", number_to_string(x));
    Ok(x as u64)
}

/// `parseArray`'s check: `v` is an array.
pub fn verify_array(v: &serde_json::Value) -> anyhow::Result<&Vec<serde_json::Value>> {
    v.as_array().ok_or_else(|| anyhow::anyhow!("Expected array, but received: {v}."))
}

/// A JSON array of `xs`, each as [`number`] writes it.
pub fn numbers<T: Copy + Into<f64>>(xs: &[T]) -> serde_json::Value {
    serde_json::Value::Array(xs.iter().map(|&x| number(x.into())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_matches_math_round() {
        for (x, want) in [
            (2.5, 3.0),
            (-2.5, -2.0),
            (-0.5, 0.0),
            (0.49999999999999994, 0.0),
            (1.4, 1.0),
            (-1.6, -2.0),
        ] {
            assert_eq!(round(x), want, "Math.round({x})");
        }
    }

    #[test]
    fn to_uint32_wraps() {
        assert_eq!(to_uint32(-1.0), 4_294_967_295);
        assert_eq!(to_uint32(4_294_967_301.0), 5);
        assert_eq!(to_uint32(-3.9), 4_294_967_293);
        assert_eq!(to_uint32(f64::NAN), 0);
    }
}

#[cfg(test)]
mod parse_float_tests {
    use super::*;

    #[test]
    fn parse_float_reads_prefixes_like_js() {
        for (input, want) in [
            ("12px", 12.0),
            ("  -1e3x", -1000.0),
            ("0x10", 0.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("-.5e-1", -0.05),
            ("1e", 1.0),
            ("+Infinity", f64::INFINITY),
            ("-Infinityx", f64::NEG_INFINITY),
            ("\u{A0}7", 7.0),
        ] {
            assert_eq!(parse_float(input), want, "parseFloat({input:?})");
        }
        for input in ["", ".", "-", "e5", "abc", "infinity"] {
            assert!(parse_float(input).is_nan(), "parseFloat({input:?})");
        }
    }

    #[test]
    fn to_string_matches_js_string() {
        use serde_json::json;
        assert_eq!(to_string(&json!([1, 2])), "1,2");
        assert_eq!(to_string(&json!(1.5)), "1.5");
        assert_eq!(to_string(&json!(3)), "3");
        assert_eq!(to_string(&json!([null, 2])), ",2");
        assert_eq!(parse_float(&to_string(&json!([7]))), 7.0);
    }

    #[test]
    fn verify_int_reads_as_parse_int_does() {
        use serde_json::json;
        assert_eq!(verify_int(&json!(2.5)).unwrap(), 2.0);
        assert_eq!(verify_int(&json!("7")).unwrap(), 7.0);
        assert_eq!(verify_int(&json!("0x1f")).unwrap(), 0.0);
        assert!(verify_int(&json!("x")).is_err());
        assert!(verify_int(&json!(null)).is_err());
        assert_eq!(verify_positive_int(&json!(3)).unwrap(), 3);
        assert!(verify_positive_int(&json!(0)).is_err());
    }

    #[test]
    fn verify_float_accepts_numeric_strings() {
        use serde_json::json;
        assert_eq!(verify_float(&json!("2.5")).unwrap(), 2.5);
        assert!(verify_float(&json!(true)).is_err());
        assert!(verify_finite_positive_float(&json!(0)).is_err());
        assert!(verify_finite_float(&json!("Infinity")).is_err());
    }
}
