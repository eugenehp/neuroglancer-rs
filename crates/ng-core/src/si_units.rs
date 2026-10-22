//! SI units and prefixes. Port of `src/util/si_units.ts`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiPrefix {
    pub prefix: &'static str,
    pub exponent: i32,
    pub long_prefix: &'static str,
}

const fn p(prefix: &'static str, exponent: i32, long_prefix: &'static str) -> SiPrefix {
    SiPrefix {
        prefix,
        exponent,
        long_prefix,
    }
}

/// Prefixes used when formatting, largest first.
pub const PREFERRED_SI_PREFIXES: &[SiPrefix] = &[
    p("Y", 24, "yotta"),
    p("Z", 21, "zetta"),
    p("E", 18, "exa"),
    p("P", 15, "peta"),
    p("T", 12, "tera"),
    p("G", 9, "giga"),
    p("M", 6, "mega"),
    p("k", 3, "kilo"),
    p("", 0, ""),
    p("m", -3, "milli"),
    p("µ", -6, "micro"),
    p("n", -9, "nano"),
    p("p", -12, "pico"),
    p("f", -15, "femto"),
    p("a", -18, "atto"),
    p("z", -21, "zepto"),
    p("y", -24, "yocto"),
];

/// Accepted when parsing but never chosen when formatting.
const EXTRA_SI_PREFIXES: &[SiPrefix] = &[
    p("h", 2, "hecto"),
    p("da", 1, "deca"),
    p("d", -1, "deci"),
    p("c", -2, "centi"),
];

/// The base units neuroglancer accepts in a coordinate space.
pub const BASE_UNITS: &[&str] = &["m", "s", "Hz", "rad/s"];

/// A unit string split into its base unit and power-of-ten prefix, e.g.
/// `"nm"` → `("m", -9)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedUnit {
    pub unit: &'static str,
    pub exponent: i32,
}

/// Looks up a unit string the way neuroglancer's `supportedUnits` map does.
/// `""` is unitless. `"u"` is accepted for micro alongside `"µ"`.
pub fn parse_unit(s: &str) -> Option<ParsedUnit> {
    if s.is_empty() {
        return Some(ParsedUnit {
            unit: "",
            exponent: 0,
        });
    }
    let alternatives = [p("u", -6, "")];
    let prefixes = alternatives
        .iter()
        .chain(PREFERRED_SI_PREFIXES)
        .chain(EXTRA_SI_PREFIXES);
    // Several (prefix, unit) pairs can spell the same string only if a prefix
    // is also a unit, which none are; the first match is the only match.
    for prefix in prefixes {
        if let Some(rest) = s.strip_prefix(prefix.prefix) {
            if let Some(&unit) = BASE_UNITS.iter().find(|&&u| u == rest) {
                return Some(ParsedUnit {
                    unit,
                    exponent: prefix.exponent,
                });
            }
        }
    }
    None
}

/// `parseScale`: `"4nm"` → `(4e-9, "m")`, `""` → `(1, "")`; `None` for
/// anything else that is not a positive finite number with an optional
/// supported unit.
pub fn parse_scale(s: &str) -> Option<(f64, &'static str)> {
    if s.is_empty() {
        return Some((1.0, ""));
    }
    // `/^((?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)?([µa-zA-Z]+)?$/`.
    let split = s.find(|c: char| c == 'µ' || c.is_ascii_alphabetic() && c != 'e' && c != 'E').unwrap_or(s.len());
    // An exponent's `e` belongs to the number only if digits follow it.
    let (mut number, mut unit) = s.split_at(split);
    if let Some(e) = number.find(['e', 'E']) {
        let after = &number[e + 1..];
        let digits = after.strip_prefix(['+', '-']).unwrap_or(after);
        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            (number, unit) = s.split_at(e);
        }
    }
    let valid_number = |n: &str| {
        let mantissa_end = n.find(['e', 'E']).unwrap_or(n.len());
        let (m, exp) = n.split_at(mantissa_end);
        let (int, frac) = match m.split_once('.') {
            Some((i, f)) => (i, Some(f)),
            None => (m, None),
        };
        let int_ok = int == "0" || (!int.is_empty() && !int.starts_with('0') && int.chars().all(|c| c.is_ascii_digit()));
        let frac_ok = frac.is_none_or(|f| !f.is_empty() && f.chars().all(|c| c.is_ascii_digit()));
        let exp_ok = exp.is_empty() || {
            let d = exp[1..].strip_prefix(['+', '-']).unwrap_or(&exp[1..]);
            !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())
        };
        int_ok && frac_ok && exp_ok
    };
    if !number.is_empty() && !valid_number(number) {
        return None;
    }
    if !unit.chars().all(|c| c == 'µ' || c.is_ascii_alphabetic()) {
        return None;
    }
    let mut scale = if number.is_empty() { 1.0 } else { number.parse::<f64>().ok()? };
    let mut base = "";
    if !unit.is_empty() {
        let parsed = parse_unit(unit)?;
        base = parsed.unit;
        scale = scale_by_exp10(scale, parsed.exponent);
    }
    (scale > 0.0 && scale.is_finite()).then_some((scale, base))
}

/// Returns `scale * 10^exponent`, dividing for negative exponents to reduce
/// loss of precision. Matches `scaleByExp10` bit for bit.
pub fn scale_by_exp10(scale: f64, exponent: i32) -> f64 {
    if exponent >= 0 {
        scale * pow10(exponent)
    } else {
        scale / pow10(-exponent)
    }
}

fn pow10(n: i32) -> f64 {
    exp10(n)
}

/// `10 ** n` as V8 computes it: the correctly rounded value (checked for
/// -24..=24, every SI prefix). `powi` multiplies repeatedly and can be an ulp
/// off once `10^n` is not exact.
pub fn exp10(n: i32) -> f64 {
    const POS: [f64; 25] = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
        1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24,
    ];
    const NEG: [f64; 25] = [
        1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
        1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24,
    ];
    let table = if n >= 0 { &POS } else { &NEG };
    match table.get(n.unsigned_abs() as usize) {
        Some(&x) => x,
        None => 10f64.powi(n),
    }
}

/// Every SI prefix, preferred and extra (`allSiPrefixes`).
pub fn all_si_prefixes() -> impl Iterator<Item = &'static SiPrefix> {
    PREFERRED_SI_PREFIXES.iter().chain(EXTRA_SI_PREFIXES)
}

/// `pickSiPrefix`: the first preferred prefix whose exponent is at most
/// `log10(x)`, else the smallest.
pub fn pick_si_prefix(x: f64) -> &'static SiPrefix {
    let e = crate::js::log10(x);
    // `binarySearchLowerBound` over a predicate true from some index on.
    let n = PREFERRED_SI_PREFIXES.len();
    let (mut begin, mut count) = (0usize, n);
    while count > 0 {
        let step = count / 2;
        let i = begin + step;
        if PREFERRED_SI_PREFIXES[i].exponent as f64 <= e {
            count = step;
        } else {
            begin = i + 1;
            count -= step + 1;
        }
    }
    &PREFERRED_SI_PREFIXES[begin.min(n - 1)]
}

/// A scale formatted for display: `(scale, prefix, unit)`, as
/// `formatScaleWithUnit(scale, unit)` with its defaults (precision 6, a
/// scale of 1 elided): `4e-9 m` → `("4", "n", "m")`.
pub fn format_scale_with_unit(scale: f64, unit: &str) -> (String, &'static str, String) {
    format_scale_with_unit_elide(scale, unit, true)
}

/// `formatScaleWithUnitAsString(scale, unit, {elide1})`.
pub fn format_scale_with_unit_as_string(scale: f64, unit: &str, elide1: bool) -> String {
    let (s, p, u) = format_scale_with_unit_elide(scale, unit, elide1);
    format!("{s}{p}{u}")
}

/// [`format_scale_with_unit`] with `elide1` as given.
pub fn format_scale_with_unit_elide(scale: f64, unit: &str, elide1: bool) -> (String, &'static str, String) {
    format_scale_with_unit_precision(scale, unit, elide1, 6)
}

/// `formatScaleWithUnitAsString(scale, unit, {precision, elide1})`.
pub fn format_scale_with_unit_as_string_precision(scale: f64, unit: &str, elide1: bool, precision: usize) -> String {
    let (s, p, u) = format_scale_with_unit_precision(scale, unit, elide1, precision);
    format!("{s}{p}{u}")
}

/// `formatScaleWithUnit(scale, unit, {precision, elide1})` (`precision` 0:
/// the number as JavaScript writes it).
pub fn format_scale_with_unit_precision(scale: f64, unit: &str, elide1: bool, precision: usize) -> (String, &'static str, String) {
    let (mut adjusted, mut prefix) = (scale, "");
    if !unit.is_empty() {
        let p = pick_si_prefix(scale);
        prefix = p.prefix;
        adjusted = scale_by_exp10(scale, -p.exponent);
    }
    if elide1 && adjusted == 1.0 {
        return (String::new(), prefix, unit.to_owned());
    }
    if precision == 0 {
        return (crate::js::number_to_string(adjusted), prefix, unit.to_owned());
    }
    let s = if !(1.0..1000.0).contains(&adjusted) { crate::js::to_precision(adjusted, precision) } else { crate::js::to_fixed(adjusted, precision) };
    let (num, exp) = match s.find('e') {
        Some(i) => s.split_at(i),
        None => (s.as_str(), ""),
    };
    // `/.*\.(?:[0-9]*[1-9])?(0+)$/`: trailing zeros after the point go.
    let num = if num.contains('.') && num.ends_with('0') { num.trim_end_matches('0').trim_end_matches('.') } else { num };
    (format!("{num}{exp}"), prefix, unit.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scales_as_the_position_widget_does() {
        assert_eq!(parse_scale("4nm"), Some((4e-9, "m")));
        assert_eq!(parse_scale(""), Some((1.0, "")));
        assert_eq!(parse_scale("nm"), Some((1e-9, "m")));
        assert_eq!(parse_scale("2.5e3um"), Some((2.5e-3, "m")));
        assert_eq!(parse_scale("10ms"), Some((0.01, "s")));
        assert_eq!(parse_scale("3"), Some((3.0, "")));
        for bad in ["0", "-4nm", "04nm", "4 nm", "4xx", "1.nm", "4e"] {
            assert_eq!(parse_scale(bad), None, "{bad}");
        }
    }

    #[test]
    fn formats_scales_as_the_position_widget_does() {
        let f = |s, u| {
            let (a, b, c) = format_scale_with_unit(s, u);
            format!("{a}{b}{c}")
        };
        assert_eq!(f(4e-9, "m"), "4nm");
        assert_eq!(f(40e-9, "m"), "40nm");
        assert_eq!(f(1e-9, "m"), "nm");
        assert_eq!(f(1.0, ""), "");
        assert_eq!(f(0.5, ""), "0.5");
        assert_eq!(f(8e-9, "s"), "8ns");
        assert_eq!(f(1.5e-6, "m"), "1.5µm");
        assert_eq!(f(2500.0, ""), "2500");
        assert_eq!(f(1234567.0, ""), "1.23457e+6");
    }

    #[test]
    fn parses_prefixed_units() {
        assert_eq!(parse_unit("nm"), Some(ParsedUnit { unit: "m", exponent: -9 }));
        assert_eq!(parse_unit("um"), Some(ParsedUnit { unit: "m", exponent: -6 }));
        assert_eq!(parse_unit("µm"), Some(ParsedUnit { unit: "m", exponent: -6 }));
        assert_eq!(parse_unit("m"), Some(ParsedUnit { unit: "m", exponent: 0 }));
        assert_eq!(parse_unit("ms"), Some(ParsedUnit { unit: "s", exponent: -3 }));
        assert_eq!(parse_unit("dam"), Some(ParsedUnit { unit: "m", exponent: 1 }));
        assert_eq!(parse_unit("krad/s"), Some(ParsedUnit { unit: "rad/s", exponent: 3 }));
        assert_eq!(parse_unit(""), Some(ParsedUnit { unit: "", exponent: 0 }));
        assert_eq!(parse_unit("parsec"), None);
        assert_eq!(parse_unit("nmm"), None);
    }

    #[test]
    fn scale_matches_js() {
        // 4 * 10**-9 computed as 4 / 1e9, which JS prints as 4e-9.
        assert_eq!(scale_by_exp10(4.0, -9), 4e-9);
        assert_eq!(scale_by_exp10(4.0, 3), 4000.0);
    }
}
