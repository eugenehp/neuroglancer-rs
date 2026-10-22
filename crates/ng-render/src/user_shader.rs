//! User image shaders (`ImageRenderLayer`, `src/webgl/shader_ui_controls.ts`):
//! GLSL with `#uicontrol` directives, run per voxel of a cross-section.
//!
//! The directives are parsed as neuroglancer parses them (`slider`, `color`,
//! `checkbox`, `invlerp`) and become what neuroglancer's shader builder makes
//! of them: `#define`s, uniforms and invlerp functions. The user's code is
//! wrapped with the image layer's helpers (`getDataValue`, `toNormalized`,
//! `emitRGB` …, the colormaps) into a GLSL fragment shader, which naga
//! translates to WGSL. That WGSL becomes the slice pipeline's
//! `slice_color`: uniforms are read from its lookup texture (the bits of each
//! value, [`control_values`]) into a private array the GLSL indexes, then the
//! user's `main` runs and what it emitted is the colour.

use ng_core::DataType;
use serde_json::Value;
use threers::volume::VolumeSample;

use crate::image::{lerp_params, Interval};
use crate::transfer_function::{lookup_table_size, parse_control_points, SortedControlPoints, TransferFunction};

/// A `#uicontrol`'s kind and defaults.
#[derive(Debug, Clone, PartialEq)]
pub enum Control {
    /// `slider`: a number (`float`, `int` or `uint`).
    Slider { value_type: SliderType, min: f64, max: f64, step: f64, default: f64 },
    /// `color` (`vec3`).
    Color { default: [f32; 3], default_string: String },
    /// `checkbox` (`bool`): compiled in.
    Checkbox { default: bool },
    /// `invlerp` over the image's data.
    Invlerp { clamp: bool, range: Interval, window: [f64; 2], channel: Vec<u32> },
    /// `select`: one of `options` (a dropdown).
    Select { value_type: SelectType, options: Vec<SelectOption>, default: SelectValue },
    /// `transferFunction`: colours by data value, from control points.
    TransferFunction { default: TransferFunction },
}

/// A `select` control's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectType {
    String,
    Int,
    Uint,
    Float,
}

/// `ShaderSelectValue`.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectValue {
    String(String),
    Number(f64),
}

impl SelectValue {
    /// `formatShaderSelectValue`.
    pub fn text(&self) -> String {
        match self {
            SelectValue::String(s) => s.clone(),
            SelectValue::Number(n) => ng_core::js::number_to_string(*n),
        }
    }

    /// The value as JSON (`shaderControls`).
    pub fn json(&self) -> Value {
        match self {
            SelectValue::String(s) => Value::String(s.clone()),
            // Integers as integers, as JavaScript writes them.
            SelectValue::Number(n) if n.fract() == 0.0 && n.abs() < 9.0e15 => Value::from(*n as i64),
            SelectValue::Number(n) => serde_json::Number::from_f64(*n).map_or(Value::Null, Value::Number),
        }
    }
}

/// `ShaderSelectOption`.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectOption {
    pub value: SelectValue,
    pub label: Option<String>,
}

impl SelectOption {
    /// `getShaderSelectOptionLabel`: the label, and the value when they
    /// differ.
    pub fn display(&self) -> String {
        let text = self.value.text();
        match &self.label {
            Some(l) if *l != text => format!("{l} ({text})"),
            _ => text,
        }
    }
}

impl SelectType {
    fn parse(s: &str) -> Option<SelectType> {
        Some(match s {
            "string_t" => SelectType::String,
            "int" => SelectType::Int,
            "uint" => SelectType::Uint,
            "float" => SelectType::Float,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            SelectType::String => "string_t",
            SelectType::Int => "int",
            SelectType::Uint => "uint",
            SelectType::Float => "float",
        }
    }

    /// `verifyShaderSelectValue`.
    pub fn verify(self, v: &Value) -> Result<SelectValue, String> {
        match self {
            SelectType::String => v.as_str().map(|s| SelectValue::String(s.to_owned())).ok_or_else(|| format!("Expected string, but received: {v}")),
            SelectType::Int => v.as_f64().filter(|x| x.fract() == 0.0).map(SelectValue::Number).ok_or_else(|| format!("Expected integer, but received: {v}")),
            SelectType::Uint => v.as_f64().filter(|x| x.fract() == 0.0 && *x >= 0.0).map(SelectValue::Number).ok_or_else(|| format!("Expected non-negative integer, but received: {v}")),
            SelectType::Float => v.as_f64().filter(|x| x.is_finite()).map(SelectValue::Number).ok_or_else(|| format!("Expected finite floating-point number, but received: {v}")),
        }
    }
}

/// `parseSelectOptions`.
fn parse_select_options(t: SelectType, value: &Value) -> Result<Vec<SelectOption>, String> {
    let option = |v: &Value, label: Option<&str>| {
        t.verify(v).map(|value| SelectOption { value, label: label.map(str::to_owned) }).map_err(|m| match label {
            None => format!("Invalid option value: {m}"),
            Some(l) => format!("Invalid option value for {}: {m}", serde_json::to_string(l).unwrap()),
        })
    };
    let options = match value {
        Value::Array(a) => a.iter().map(|v| option(v, None)).collect::<Result<Vec<_>, _>>()?,
        Value::Object(o) => o.iter().map(|(l, v)| option(v, Some(l))).collect::<Result<Vec<_>, _>>()?,
        _ => {
            let n = t.name();
            return Err(format!("Expected options argument to be an array of {n} values or an object mapping labels to {n} values"));
        }
    };
    if options.is_empty() {
        return Err("Expected options argument to contain at least one option".into());
    }
    for (i, o) in options.iter().enumerate() {
        if options[..i].iter().any(|p| p.value == o.value) {
            return Err(format!("Duplicate option value: {}", o.value.json()));
        }
    }
    Ok(options)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliderType {
    Float,
    Int,
    Uint,
}

impl SliderType {
    pub(crate) fn glsl(self) -> &'static str {
        match self {
            SliderType::Float => "float",
            SliderType::Int => "int",
            SliderType::Uint => "uint",
        }
    }
}

/// A problem in a directive, by line (from 0).
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderError {
    pub line: usize,
    pub message: String,
}

/// `ShaderControlsParseResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedShader {
    /// The code with comments blanked and directives removed.
    pub code: String,
    pub controls: Vec<(String, Control)>,
    /// The code's string literals, by id less one (`stringLiteralIds`).
    pub string_literals: Vec<String>,
    pub errors: Vec<ShaderError>,
    /// The image's channel rank (`channelRank`): how many indices
    /// `getDataValue` takes, and a `channel` selects.
    pub channel_rank: usize,
}

/// `stripComments`: comments blanked (newlines kept), string literals kept.
pub fn strip_comments(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                out.push(if chars[i].is_whitespace() { chars[i] } else { ' ' });
                i += 1;
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            let start = i;
            let mut end = None;
            let mut j = i + 2;
            while j + 1 < chars.len() {
                if chars[j] == '*' && chars[j + 1] == '/' {
                    end = Some(j + 2);
                    break;
                }
                j += 1;
            }
            // An unclosed comment is left (`/\*(?:.|\n)*?\*\/` needs its end).
            let Some(end) = end else {
                out.push(c);
                i += 1;
                continue;
            };
            for &ch in &chars[start..end] {
                out.push(if ch.is_whitespace() { ch } else { ' ' });
            }
            i = end;
        } else if c == '"' || c == '\'' {
            out.push(c);
            i += 1;
            while i < chars.len() && chars[i] != c {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    out.push(chars[i]);
                    i += 1;
                }
                out.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                out.push(chars[i]);
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// `matchDirectiveParameterValue`: how much of `input` a value spans
/// (`None` for none or unbalanced brackets).
fn match_value(input: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut rest = input;
    loop {
        if rest.is_empty() {
            break;
        }
        let token_len = token(rest)?;
        let c = rest.as_bytes()[0] as char;
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            ',' if depth == 0 => break,
            ',' => {}
            _ if depth == 0 => {
                rest = &rest[token_len..];
                break;
            }
            _ => {}
        }
        rest = &rest[token_len..];
    }
    (depth == 0).then_some(input.len() - rest.len())
}

/// A value token's length: a number, a string, `true`, `false`, `:`,
/// whitespace, `,`, a bracket.
fn token(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let c = b[0] as char;
    if c.is_whitespace() {
        return Some(s.len() - s.trim_start().len());
    }
    if matches!(c, ':' | ',' | '[' | ']' | '{' | '}') {
        return Some(1);
    }
    for word in ["true", "false"] {
        if s.starts_with(word) {
            return Some(word.len());
        }
    }
    if c == '"' {
        let mut i = 1;
        while i < b.len() {
            match b[i] {
                b'\\' => i += 2,
                b'"' => return Some(i + 1),
                _ => i += 1,
            }
        }
        return None;
    }
    // `-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?`.
    let mut i = 0;
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(d) if d.is_ascii_digit() => {
            while b.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        _ => return None,
    }
    if b.get(i) == Some(&b'.') && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
        i += 1;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(b.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        if b.get(j).is_some_and(u8::is_ascii_digit) {
            while b.get(j).is_some_and(u8::is_ascii_digit) {
                j += 1;
            }
            i = j;
        }
    }
    Some(i)
}

/// `parseDirectiveParameters`: `name=value, …` (values as JSON).
pub fn parse_directive_parameters(input: Option<&str>) -> (Vec<(String, Value)>, Vec<String>) {
    let mut errors = vec![];
    let mut parameters: Vec<(String, Value)> = vec![];
    let Some(mut input) = input else { return (parameters, errors) };
    const SYNTAX: &str = "Invalid #uicontrol parameter syntax, expected: <param>=<value>, ...";
    loop {
        input = input.trim();
        if input.is_empty() {
            break;
        }
        // `^([_a-z][_a-zA-Z0-9]*)[ \t]*=`.
        let name_len = input.char_indices().take_while(|&(i, c)| if i == 0 { c == '_' || c.is_ascii_lowercase() } else { c == '_' || c.is_ascii_alphanumeric() }).count();
        let after = input[name_len..].trim_start_matches([' ', '\t']);
        if name_len == 0 || !after.starts_with('=') {
            errors.push(SYNTAX.into());
            break;
        }
        let name = input[..name_len].to_owned();
        input = &after[1..];
        let Some(len) = match_value(input).filter(|&l| l > 0) else {
            errors.push(SYNTAX.into());
            break;
        };
        let value: Value = match serde_json::from_str(input[..len].trim()) {
            Ok(v) => v,
            Err(_) => {
                errors.push(format!("Invalid #uicontrol parameter value for {name}: undefined"));
                break;
            }
        };
        if parameters.iter().any(|(n, _)| *n == name) {
            errors.push(format!("Duplicate #uicontrol parameter: {name}"));
        } else {
            parameters.push((name, value));
        }
        input = input[len..].trim();
        if !input.is_empty() && !input.starts_with(',') {
            errors.push(SYNTAX.into());
        }
        input = input.get(1..).unwrap_or("");
    }
    (parameters, errors)
}

/// `parseRGBColorSpecification`: `#rgb`, `#rrggbb` or a CSS name (the few
/// shaders use).
pub fn parse_color(s: &str) -> Option<[f32; 3]> {
    let named = match s {
        "white" => Some([1.0, 1.0, 1.0]),
        "black" => Some([0.0, 0.0, 0.0]),
        "red" => Some([1.0, 0.0, 0.0]),
        "green" => Some([0.0, 128.0 / 255.0, 0.0]),
        "lime" => Some([0.0, 1.0, 0.0]),
        "blue" => Some([0.0, 0.0, 1.0]),
        "yellow" => Some([1.0, 1.0, 0.0]),
        "cyan" | "aqua" => Some([0.0, 1.0, 1.0]),
        "magenta" | "fuchsia" => Some([1.0, 0.0, 1.0]),
        "gray" | "grey" => Some([128.0 / 255.0; 3]),
        "orange" => Some([1.0, 165.0 / 255.0, 0.0]),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let hex = s.strip_prefix('#')?;
    let digit = |c: char| c.to_digit(16).map(|d| d as f32);
    let chars: Vec<char> = hex.chars().collect();
    match chars.len() {
        3 => Some([digit(chars[0])? * 17.0 / 255.0, digit(chars[1])? * 17.0 / 255.0, digit(chars[2])? * 17.0 / 255.0]),
        6 => Some([
            (digit(chars[0])? * 16.0 + digit(chars[1])?) / 255.0,
            (digit(chars[2])? * 16.0 + digit(chars[3])?) / 255.0,
            (digit(chars[4])? * 16.0 + digit(chars[5])?) / 255.0,
        ]),
        _ => None,
    }
}

/// A data type's interval from JSON (`parseDataTypeInterval`): two numbers,
/// or for uint64 two decimal strings (or numbers).
pub fn parse_interval(v: &Value, t: DataType) -> Option<Interval> {
    let a = v.as_array().filter(|a| a.len() == 2)?;
    if t == DataType::Uint64 {
        let parse = |x: &Value| match x {
            Value::String(s) => s.parse::<u64>().ok(),
            Value::Number(n) => n.as_u64(),
            _ => None,
        };
        return Some(Interval::Uint64([parse(&a[0])?, parse(&a[1])?]));
    }
    Some(Interval::Number([a[0].as_f64()?, a[1].as_f64()?]))
}

/// `normalizeDataTypeInterval`: the interval low to high.
fn normalized(i: Interval) -> [f64; 2] {
    let [a, b] = i.as_f64();
    [a.min(b), a.max(b)]
}

/// `parseInvlerpChannel`: a number, or an array of `rank` non-negative
/// integers.
pub fn parse_channel(value: &Value, rank: usize) -> Result<Vec<u32>, String> {
    let items: Vec<Value> = match value {
        Value::Number(_) => vec![value.clone()],
        Value::Array(a) => a.clone(),
        _ => return Err(format!("Expected array, but received: {value}")),
    };
    if items.len() != rank {
        return Err(format!("Expected array of length {rank}, but received: {value}"));
    }
    items
        .iter()
        .map(|x| match x.as_f64() {
            Some(f) if f >= 0.0 && f.fract() == 0.0 && f < 4294967296.0 => Ok(f as u32),
            _ => Err(format!("Expected non-negative integer, but received: {x}")),
        })
        .collect()
}

fn parse_control(type_name: &str, control: &str, parameters: &[(String, Value)], t: DataType, channel_rank: usize) -> Result<Control, Vec<String>> {
    let mut errors = vec![];
    match control {
        "slider" => {
            let value_type = match type_name {
                "float" => SliderType::Float,
                "int" => SliderType::Int,
                "uint" => SliderType::Uint,
                _ => {
                    errors.push("type must be float, int, or uint".to_owned());
                    SliderType::Float
                }
            };
            let (mut min, mut max, mut step, mut default) = (None, None, None, None);
            for (key, value) in parameters {
                let mut get = || -> Option<f64> {
                    let Some(v) = value.as_f64() else {
                        errors.push(format!("Expected {key} argument to be a number"));
                        return None;
                    };
                    if value_type != SliderType::Float {
                        if v.fract() != 0.0 {
                            errors.push(format!("Expected {key} argument to be an integer"));
                        }
                        if value_type == SliderType::Uint && v < 0.0 {
                            errors.push(format!("Expected {key} argument to be an unsigned integer"));
                        }
                    }
                    Some(v)
                };
                match key.as_str() {
                    "min" => min = get(),
                    "max" => max = get(),
                    "default" => default = get(),
                    "step" => step = get(),
                    _ => errors.push(format!("Invalid parameter: {key}")),
                }
            }
            if min.is_none() {
                errors.push("min must be specified".into());
            }
            if max.is_none() {
                errors.push("max must be specified".into());
            }
            if let (Some(lo), Some(hi)) = (min, max) {
                if lo > hi {
                    errors.push("min must be less than max".into());
                }
                step = Some(step.unwrap_or(if value_type == SliderType::Float { (hi - lo) / 100.0 } else { 1.0 }));
                match default {
                    Some(d) if d < lo || d > hi => errors.push("default must be within valid range".into()),
                    Some(_) => {}
                    None => default = Some(if value_type == SliderType::Float { (lo + hi) / 2.0 } else { lo }),
                }
            }
            if !errors.is_empty() {
                return Err(errors);
            }
            Ok(Control::Slider { value_type, min: min.unwrap(), max: max.unwrap(), step: step.unwrap(), default: default.unwrap() })
        }
        "checkbox" => {
            if type_name != "bool" {
                errors.push("type must be bool".into());
            }
            let mut default = false;
            for (key, value) in parameters {
                match (key.as_str(), value) {
                    ("default", Value::Bool(b)) => default = *b,
                    ("default", _) => errors.push(format!("Expected {key} argument to be a boolean")),
                    _ => errors.push(format!("Invalid parameter: {key}")),
                }
            }
            if errors.is_empty() {
                Ok(Control::Checkbox { default })
            } else {
                Err(errors)
            }
        }
        "color" => {
            if type_name != "vec3" {
                errors.push("type must be vec3".into());
            }
            let mut default_string = "white".to_owned();
            for (key, value) in parameters {
                match (key.as_str(), value) {
                    ("default", Value::String(s)) => default_string = s.clone(),
                    ("default", _) => errors.push("Expected default argument to be a string".into()),
                    _ => errors.push(format!("Invalid parameter: {key}")),
                }
            }
            let default = parse_color(&default_string);
            if default.is_none() && errors.is_empty() {
                errors.push(format!("Invalid color: {default_string}"));
            }
            if errors.is_empty() {
                Ok(Control::Color { default: default.unwrap(), default_string })
            } else {
                Err(errors)
            }
        }
        "invlerp" => {
            if type_name != "invlerp" {
                errors.push("type must be invlerp".into());
            }
            let mut clamp = true;
            let mut range = crate::image::default_data_type_range(t);
            let mut window = None;
            let mut channel = vec![0; channel_rank];
            for (key, value) in parameters {
                match key.as_str() {
                    "range" => match parse_interval(value, t) {
                        Some(r) => range = r,
                        None => errors.push(format!("Invalid range value: {value}")),
                    },
                    "window" => match parse_interval(value, t) {
                        Some(w) if normalized(w)[0] < normalized(w)[1] => window = Some(normalized(w)),
                        _ => errors.push(format!("Invalid window value: {value}")),
                    },
                    "clamp" => match value {
                        Value::Bool(b) => clamp = *b,
                        _ => errors.push(format!("Invalid clamp value: {value}")),
                    },
                    "channel" => match parse_channel(value, channel_rank) {
                        Ok(c) => channel = c,
                        Err(m) => errors.push(format!("Invalid channel value: {m}")),
                    },
                    _ => errors.push(format!("Invalid parameter: {key}")),
                }
            }
            if errors.is_empty() {
                Ok(Control::Invlerp { clamp, range, window: window.unwrap_or_else(|| normalized(range)), channel })
            } else {
                Err(errors)
            }
        }
        "select" => {
            let value_type = SelectType::parse(type_name);
            if value_type.is_none() {
                errors.push("type must be one of string_t, int, uint, or float".into());
            }
            let (mut options, mut default) = (None, None);
            for (key, value) in parameters {
                match (key.as_str(), value_type) {
                    ("options", Some(vt)) => match parse_select_options(vt, value) {
                        Ok(o) => options = Some(o),
                        Err(e) => errors.push(e),
                    },
                    ("default", Some(vt)) => match vt.verify(value) {
                        Ok(v) => default = Some(v),
                        Err(m) => errors.push(format!("Invalid default value: {m}")),
                    },
                    ("options" | "default", None) => {}
                    _ => errors.push(format!("Invalid parameter: {key}")),
                }
            }
            if !parameters.iter().any(|(k, _)| k == "options") {
                errors.push("options must be specified".into());
            }
            let Some(options) = options else { return Err(errors) };
            let mut resolved = options[0].value.clone();
            if let Some(d) = default {
                if options.iter().any(|o| o.value == d) {
                    resolved = d;
                } else {
                    errors.push(format!("default value {} must match one of the options", d.json()));
                }
            }
            if !errors.is_empty() {
                return Err(errors);
            }
            Ok(Control::Select { value_type: value_type.unwrap(), options, default: resolved })
        }
        "transferFunction" => {
            if type_name != "transferFunction" {
                errors.push("type must be transferFunction".into());
            }
            let mut default_color = [1.0, 1.0, 1.0];
            let mut window = None;
            let mut points = SortedControlPoints::new(vec![], t);
            let mut channel = vec![0; channel_rank];
            for (key, value) in parameters {
                let result = match key.as_str() {
                    "channel" => parse_channel(value, channel_rank).map(|c| channel = c),
                    "defaultColor" => value.as_str().and_then(parse_color).map(|c| default_color = c).ok_or_else(|| format!("Invalid color: {value}")),
                    "window" => match parse_interval(value, t) {
                        Some(w) if w.as_f64()[0] <= w.as_f64()[1] => {
                            window = Some(w);
                            Ok(())
                        }
                        Some(w) => Err(format!("Invalid interval: [{}, {}]", w.as_f64()[0], w.as_f64()[1])),
                        None => Err(format!("Invalid window: {value}")),
                    },
                    "controlPoints" => parse_control_points(value, t).map(|p| points = p),
                    _ => {
                        errors.push(format!("Invalid parameter: {key}"));
                        Ok(())
                    }
                };
                if let Err(m) = result {
                    errors.push(format!("Invalid {key} value: {m}"));
                }
            }
            if !errors.is_empty() {
                return Err(errors);
            }
            let window = window.unwrap_or(points.range);
            Ok(Control::TransferFunction { default: TransferFunction { points, window, channel, default_color } })
        }
        _ => Err(vec![format!("Invalid control type {control}")]),
    }
}

/// `parseShaderUiControls` for a single-channel image of type `t`.
pub fn parse(code: &str, t: DataType) -> ParsedShader {
    parse_with_channels(code, t, 0)
}

/// `parseShaderUiControls` for an image of type `t` with `channel_rank`
/// channel dimensions.
pub fn parse_with_channels(code: &str, t: DataType, channel_rank: usize) -> ParsedShader {
    let stripped = strip_comments(code);
    let mut out = String::with_capacity(stripped.len());
    let mut controls: Vec<(String, Control)> = vec![];
    let mut errors = vec![];
    for (line, text) in stripped.split('\n').enumerate() {
        if line > 0 {
            out.push('\n');
        }
        // `^[ \t]*#[ \t]*uicontrol[ \t]+(.*)$`.
        let directive = text.trim_start_matches([' ', '\t']).strip_prefix('#').map(|r| r.trim_start_matches([' ', '\t'])).and_then(|r| r.strip_prefix("uicontrol")).filter(|r| r.starts_with([' ', '\t']));
        let Some(inner) = directive else {
            out.push_str(text);
            continue;
        };
        let inner = inner.trim_start_matches([' ', '\t']);
        let mut error = |message: String| errors.push(ShaderError { line, message });
        // `^([_a-zA-Z][_a-zA-Z0-9]*)[ \t]+([a-z][a-zA-Z0-9_]*)(?:[ \t]+([a-z]+))?[ \t]*(?:\([ \t]*(.*)\)[ \t]*)?`.
        let ident = |s: &str, first: fn(char) -> bool| s.char_indices().take_while(|&(i, c)| if i == 0 { first(c) } else { c == '_' || c.is_ascii_alphanumeric() }).count();
        let type_len = ident(inner, |c| c == '_' || c.is_ascii_alphabetic());
        let rest = &inner[type_len..];
        let rest_trim = rest.trim_start_matches([' ', '\t']);
        let name_len = if rest_trim.len() < rest.len() { ident(rest_trim, |c| c.is_ascii_lowercase()) } else { 0 };
        if type_len == 0 || name_len == 0 {
            error("Invalid #uicontrol syntax, expected: #uicontrol <type> <name> <control>(<param>=<value>, ...)".into());
            continue;
        }
        let type_name = &inner[..type_len];
        let name = &rest_trim[..name_len];
        let mut rest = &rest_trim[name_len..];
        let mut control = type_name;
        let after = rest.trim_start_matches([' ', '\t']);
        if after.len() < rest.len() {
            let n = after.chars().take_while(char::is_ascii_lowercase).count();
            if n > 0 {
                control = &after[..n];
                rest = &after[n..];
            }
        }
        let rest = rest.trim_start_matches([' ', '\t']);
        let parameters_text = rest.strip_prefix('(').and_then(|r| r.trim_end().strip_suffix(')')).map(|r| r.trim_start_matches([' ', '\t']));
        let (parameters, inner_errors) = parse_directive_parameters(parameters_text);
        let had_errors = !inner_errors.is_empty();
        for e in inner_errors {
            error(e);
        }
        if controls.iter().any(|(n, _)| n == name) {
            error(format!("Duplicate definition for control {name}"));
        }
        if had_errors {
            continue;
        }
        match parse_control(type_name, control, &parameters, t, channel_rank) {
            Ok(c) => controls.push((name.to_owned(), c)),
            Err(es) => {
                for e in es {
                    error(e);
                }
            }
        }
    }
    let (code, string_literals) = preprocess_strings(&out);
    ParsedShader { code, controls, errors, string_literals, channel_rank }
}

/// `preprocessStrings`: each double-quoted literal as `string_t(<id>u)`,
/// ids from 1 in order of appearance.
fn preprocess_strings(code: &str) -> (String, Vec<String>) {
    let mut ids: Vec<String> = vec![];
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    while let Some(start) = rest.find('"') {
        // `"(?:\\.|[^\\"])*"`.
        let b = rest.as_bytes();
        let mut i = start + 1;
        let end = loop {
            match b.get(i) {
                None => break None,
                Some(b'\\') if i + 1 < b.len() => i += 2,
                Some(b'\\') => break None,
                Some(b'"') => break Some(i + 1),
                Some(_) => i += 1,
            }
        };
        let Some(end) = end else { break };
        out.push_str(&rest[..start]);
        let token = &rest[start..end];
        match serde_json::from_str::<String>(token) {
            Ok(value) => {
                let id = match ids.iter().position(|s| *s == value) {
                    Some(k) => k + 1,
                    None => {
                        ids.push(value);
                        ids.len()
                    }
                };
                out.push_str(&format!("string_t({id}u)"));
            }
            Err(_) => out.push_str(token),
        }
        rest = &rest[end..];
    }
    out.push_str(rest);
    (out, ids)
}

/// A control's current value (`shaderControls`, else its default).
#[derive(Debug, Clone, PartialEq)]
pub enum ControlValue {
    Number(f64),
    Color([f32; 3]),
    Bool(bool),
    Invlerp { range: Interval, window: [f64; 2], channel: Vec<u32> },
    Select(SelectValue),
    TransferFunction(TransferFunction),
}

/// The values of `parsed`'s controls from a layer's `shaderControls`.
pub fn control_values(parsed: &ParsedShader, t: DataType, json: Option<&Value>) -> Vec<ControlValue> {
    parsed
        .controls
        .iter()
        .map(|(name, control)| {
            let v = json.and_then(|j| j.get(name));
            match control {
                Control::Slider { default, min, max, .. } => ControlValue::Number(v.and_then(Value::as_f64).filter(|x| *x >= *min && *x <= *max).unwrap_or(*default)),
                Control::Color { default, .. } => ControlValue::Color(v.and_then(Value::as_str).and_then(parse_color).unwrap_or(*default)),
                Control::Checkbox { default } => ControlValue::Bool(v.and_then(Value::as_bool).unwrap_or(*default)),
                Control::Invlerp { range, window, channel, .. } => {
                    // `{range, window, channel}` or (legacy) a bare range.
                    let (r, w, c) = match v {
                        Some(Value::Object(o)) => (
                            o.get("range").and_then(|r| parse_interval(r, t)),
                            o.get("window").and_then(|w| parse_interval(w, t)).map(normalized),
                            o.get("channel").and_then(|c| parse_channel(c, channel.len()).ok()),
                        ),
                        Some(other) => (parse_interval(other, t), None, None),
                        None => (None, None, None),
                    };
                    let r = r.unwrap_or(*range);
                    ControlValue::Invlerp { range: r, window: w.unwrap_or(if v.is_some() { normalized(r) } else { *window }), channel: c.unwrap_or_else(|| channel.clone()) }
                }
                Control::TransferFunction { default } => ControlValue::TransferFunction(TransferFunction::from_json(v, t, default)),
                Control::Select { value_type, options, default } => {
                    ControlValue::Select(v.and_then(|v| value_type.verify(v).ok()).filter(|x| options.iter().any(|o| o.value == *x)).unwrap_or_else(|| default.clone()))
                }
            }
        })
        .collect()
}

/// How many floats a control takes in the uniform array.
fn width(control: &Control) -> usize {
    match control {
        Control::Slider { .. } => 1,
        Control::Color { .. } => 3,
        Control::Checkbox { .. } => 0,
        Control::Invlerp { .. } => 8,
        Control::Select { .. } => 1,
        // Its lerp parameters (the table follows all controls).
        Control::TransferFunction { .. } => 8,
    }
}

/// The uniform array's contents: the opacity, then each control's value
/// (an invlerp's as [`lerp_params`] gives them).
pub fn uniforms(parsed: &ParsedShader, t: DataType, values: &[ControlValue], opacity: f32) -> Vec<f32> {
    let mut out = vec![opacity];
    for ((_, control), value) in parsed.controls.iter().zip(values) {
        match (control, value) {
            (Control::Slider { .. }, ControlValue::Number(v)) => out.push(*v as f32),
            (Control::Color { .. }, ControlValue::Color(c)) => out.extend_from_slice(c),
            (Control::Invlerp { .. }, ControlValue::Invlerp { range, .. }) => {
                let [p0, p1] = lerp_params(t, *range);
                out.extend_from_slice(&p0);
                out.extend_from_slice(&p1);
            }
            // A string by its literal's id (0 if the code never names it).
            (Control::Select { .. }, ControlValue::Select(SelectValue::String(s))) => out.push(parsed.string_literals.iter().position(|l| l == s).map_or(0.0, |k| (k + 1) as f32)),
            (Control::Select { .. }, ControlValue::Select(SelectValue::Number(n))) => out.push(*n as f32),
            // Over the points' range (`enableTransferFunctionShader`).
            (Control::TransferFunction { .. }, ControlValue::TransferFunction(tf)) => {
                let [p0, p1] = lerp_params(t, tf.points.range);
                out.extend_from_slice(&p0);
                out.extend_from_slice(&p1);
            }
            (c, _) => out.extend(std::iter::repeat_n(0.0, width(c))),
        }
    }
    out
}

/// Where each transfer function's lookup table starts in the uniform
/// texture (after the controls), in control order.
fn lookup_table_bases(parsed: &ParsedShader, t: DataType, n: usize) -> Vec<usize> {
    let mut base = n;
    let mut out = vec![];
    for (_, control) in &parsed.controls {
        if let Control::TransferFunction { .. } = control {
            out.push(base);
            base += lookup_table_size(t);
        }
    }
    out
}

/// The uniform texture's contents: [`uniforms`] as bits, then each
/// transfer function's lookup table (RGBA bytes, little-endian).
pub fn uniform_bits(parsed: &ParsedShader, t: DataType, values: &[ControlValue], opacity: f32) -> Vec<u32> {
    let mut out: Vec<u32> = uniforms(parsed, t, values, opacity).iter().map(|v| v.to_bits()).collect();
    for ((_, control), value) in parsed.controls.iter().zip(values) {
        if let (Control::TransferFunction { .. }, ControlValue::TransferFunction(tf)) = (control, value) {
            out.extend(tf.points.lookup_table(lookup_table_size(t)).iter().map(|c| u32::from_le_bytes(*c)));
        }
    }
    out
}

/// The GLSL type `getDataValue` returns, and its helpers (`shader_lib.ts`).
fn data_type_glsl(t: DataType) -> (&'static str, String) {
    let uint64 = "struct uint64_t {\n  uvec2 value;\n};\nuint64_t toUint64(uint64_t x) { return x; }\n";
    let int_struct = |name: &str, scalar: &str, max: &str, conv: &str| {
        let normalized = if max.is_empty() { String::new() } else { format!("float toNormalized({name} x) {{ return float(x.value) / {max}; }}\n") };
        format!("struct {name} {{\n  {scalar} value;\n}};\n{scalar} toRaw({name} x) {{ return x.value; }}\n{normalized}{name} ng_value() {{ return {name}({conv}); }}\n")
    };
    match t {
        DataType::Uint8 => ("uint8_t", format!("{uint64}{}", int_struct("uint8_t", "uint", "255.0", "uint(ng_raw_f.r)"))),
        DataType::Int8 => ("int8_t", format!("{uint64}{}", int_struct("int8_t", "int", "", "int(ng_raw_f.r)"))),
        DataType::Uint16 => ("uint16_t", format!("{uint64}{}", int_struct("uint16_t", "uint", "65535.0", "uint(ng_raw_f.r)"))),
        DataType::Int16 => ("int16_t", format!("{uint64}{}", int_struct("int16_t", "int", "", "int(ng_raw_f.r)"))),
        DataType::Uint32 => ("uint32_t", format!("{uint64}{}", int_struct("uint32_t", "uint", "4294967295.0", "ng_raw_u.r"))),
        DataType::Int32 => ("int32_t", format!("{uint64}{}", int_struct("int32_t", "int", "", "int(ng_raw_u.r)"))),
        DataType::Uint64 => ("uint64_t", format!("{uint64}uint64_t ng_value() {{ return uint64_t(ng_raw_u.rg); }}\n")),
        DataType::Float32 => (
            "float",
            "float toRaw(float x) { return x; }\nfloat toNormalized(float x) { return x; }\nfloat ng_value() { return ng_raw_f.r; }\n".to_owned(),
        ),
    }
}

/// `computeInvlerp` for `t` over the uniform array at `k` (the layout of
/// [`lerp_params`], as the built-in shader reads it).
fn invlerp_glsl(t: DataType, name: &str, k: usize, clamp: bool, channel: &[u32]) -> String {
    let (ty, _) = data_type_glsl(t);
    let body = match t {
        DataType::Uint8 | DataType::Int8 | DataType::Uint16 | DataType::Int16 => format!("return (float(toRaw(x)) - ng_controls[{k}]) * ng_controls[{}];", k + 1),
        DataType::Float32 => format!("return (x - ng_controls[{k}]) * ng_controls[{}];", k + 1),
        DataType::Uint32 | DataType::Int32 => {
            let raw = if t == DataType::Int32 { "uint(x.value)" } else { "x.value" };
            let less = if t == DataType::Int32 { "x.value < int(offset)" } else { "x.value < offset" };
            format!(
                "uint offset = uint(ng_controls[{o}]) | (uint(ng_controls[{o1}]) << 16u);\n  float multiplier = ng_controls[{m}];\n  uint v = {raw};\n  uint d;\n  if ({less}) {{ d = offset - v; multiplier = -multiplier; }} else {{ d = v - offset; }}\n  d = d >> uint(ng_controls[{s}]);\n  return float(d) * multiplier;",
                o = k + 4,
                o1 = k + 5,
                m = k + 1,
                s = k + 3,
            )
        }
        DataType::Uint64 => format!(
            "uvec2 offset = uvec2(uint(ng_controls[{o}]) | (uint(ng_controls[{o1}]) << 16u), uint(ng_controls[{o2}]) | (uint(ng_controls[{o3}]) << 16u));\n  float multiplier = ng_controls[{m}];\n  uvec2 a = x.value;\n  uvec2 b = offset;\n  if (a.y < b.y || (a.y == b.y && a.x < b.x)) {{ a = offset; b = x.value; multiplier = -multiplier; }}\n  uvec2 d = a;\n  if (a.x < b.x) {{ d.y = d.y - 1u; }}\n  d = d - b;\n  uint shift = uint(ng_controls[{s}]);\n  uint low;\n  if (shift >= 32u) {{ low = d.y >> (shift - 32u); }} else if (shift == 0u) {{ low = d.x; }} else {{ low = (d.x >> shift) | (d.y << (32u - shift)); }}\n  return float(low) * multiplier;",
            o = k + 4,
            o1 = k + 5,
            o2 = k + 6,
            o3 = k + 7,
            m = k + 1,
            s = k + 3,
        ),
    };
    let clamp = if clamp { "v = clamp(v, 0.0, 1.0);" } else { "" };
    if name.is_empty() {
        return body;
    }
    format!("float ng_invlerp_{name}({ty} x) {{\n  {body}\n}}\nfloat u_shaderControl_{name}({ty} x) {{\n  float v = ng_invlerp_{name}(x);\n  {clamp}\n  ng_default_intensity = v;\n  return v;\n}}\nfloat u_shaderControl_{name}() {{ return u_shaderControl_{name}(getDataValue({channel})); }}\n#define {name} u_shaderControl_{name}\n", channel = channel_args(channel))
}

/// `string_t` (`glsl_string`, a struct of a `uint` there): an alias of
/// `uint` here, as naga does not compare structs for equality; constructing,
/// `==` and `!=` read the same, only `.value` does not.
pub(crate) const STRING_T: &str = "#define string_t uint\n";

/// `defineTransferFunctionShader` over the uniform array at `k`, its table
/// at `base` (read through `ng_lut`).
fn transfer_function_glsl(t: DataType, name: &str, k: usize, base: usize, channel: &[u32]) -> String {
    let (ty, _) = data_type_glsl(t);
    let body = invlerp_glsl(t, "", k, false, &[]);
    let end = lookup_table_size(t) - 1;
    let mut s = format!(
        "float ng_invlerp_{name}({ty} x) {{\n  {body}\n}}\nvec4 {name}_(float inputValue) {{\n  int index = clamp(int(round(inputValue * {end}.0)), 0, {end});\n  uint c = ng_lut({base} + index);\n  return vec4(float(c & 255u), float((c >> 8u) & 255u), float((c >> 16u) & 255u), float(c >> 24u)) / 255.0;\n}}\nvec4 {name}({ty} inputValue) {{\n  float v = ng_invlerp_{name}(inputValue);\n  return v < 0.0 ? vec4(0.0, 0.0, 0.0, 0.0) : {name}_(clamp(v, 0.0, 1.0));\n}}\nvec4 {name}() {{\n  return {name}(getDataValue({channel}));\n}}\n",
        channel = channel_args(channel)
    );
    if !matches!(t, DataType::Uint64 | DataType::Float32) {
        let scalar = if matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32) { "int" } else { "uint" };
        s += &format!("vec4 {name}({scalar} inputValue) {{\n  return {name}({ty}(inputValue));\n}}\n");
    }
    s
}

/// A stand-in the WGSL's `ng_lut` replaces: the uniform texture's texel.
const LUT_STUB: &str = "uint ng_lut(int i) { return uint(i) + 3735928559u; }\n";

pub(crate) const COLORMAPS: &str = r#"vec3 colormapJet(float x) {
  vec3 result;
  result.r = x < 0.89 ? ((x - 0.35) / 0.31) : (1.0 - (x - 0.89) / 0.11 * 0.5);
  result.g = x < 0.64 ? ((x - 0.125) * 4.0) : (1.0 - (x - 0.64) / 0.27);
  result.b = x < 0.34 ? (0.5 + x * 0.5 / 0.11) : (1.0 - (x - 0.34) / 0.31);
  return clamp(result, 0.0, 1.0);
}
vec3 colormapCubehelix(float x) {
  float xclamp = clamp(x, 0.0, 1.0);
  float angle = 2.0 * 3.1415926 * (4.0 / 3.0 + xclamp);
  float amp = xclamp * (1.0 - xclamp) / 2.0;
  vec3 result;
  float cosangle = cos(angle);
  float sinangle = sin(angle);
  result.r = -0.14861 * cosangle + 1.78277 * sinangle;
  result.g = -0.29227 * cosangle + -0.90649 * sinangle;
  result.b = 1.97294 * cosangle;
  result = clamp(xclamp + amp * result, 0.0, 1.0);
  return result;
}
"#;

/// A `channel` selection as `getDataValue`'s arguments (`0` without
/// channel dimensions: its ignored index).
fn channel_args(channel: &[u32]) -> String {
    if channel.is_empty() {
        return String::new();
    }
    channel.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
}

/// `defineChunkDataShaderAccess`'s `getDataValue` for channel dimensions of
/// `extents` (first fastest, as the channels are stacked): each index
/// clamped, the channel's raw value read (`ng_fetch`, which the WGSL
/// replaces with the texture read) and converted as the sampled one is.
fn channel_access_glsl(ty: &str, raw: &str, extents: &[u32]) -> String {
    let rank = extents.len();
    let raw_ty = if raw == "ng_raw_u" { "uvec4" } else { "vec4" };
    let mut s = format!("{raw_ty} ng_fetch(int i) {{ return {raw}; }}\n");
    if rank == 0 {
        s += &format!("{ty} getDataValue() {{ return ng_value(); }}\n{ty} getDataValue(int ignoredChannelIndex) {{ return ng_value(); }}\n{ty} getInterpolatedDataValue() {{ return ng_value(); }}\n{ty} getInterpolatedDataValue(int ignoredChannelIndex) {{ return ng_value(); }}\n");
        return s;
    }
    let params: Vec<String> = (0..rank).map(|k| format!("int channelIndex{k}")).collect();
    let args: Vec<String> = (0..rank).map(|k| format!("channelIndex{k}")).collect();
    let mut flat = String::from("0");
    for k in (0..rank).rev() {
        flat = format!("clamp(channelIndex{k}, 0, {}) + {} * ({flat})", extents[k].max(1) - 1, extents[k].max(1));
    }
    s += &format!(
        "{ty} getDataValue({p}) {{\n  {raw_ty} ng_saved = {raw};\n  {raw} = ng_fetch({flat});\n  {ty} ng_result = ng_value();\n  {raw} = ng_saved;\n  return ng_result;\n}}\n{ty} getInterpolatedDataValue({p}) {{ return getDataValue({a}); }}\n",
        p = params.join(", "),
        a = args.join(", ")
    );
    if rank == 1 {
        s += &format!("{ty} getDataValue() {{ return getDataValue(0); }}\n{ty} getInterpolatedDataValue() {{ return getDataValue(0); }}\n");
    }
    s
}

/// The GLSL fragment shader for `parsed` over data of type `t`, with
/// checkboxes at `values` (compiled in, as `#define`s).
pub fn glsl(t: DataType, parsed: &ParsedShader, values: &[ControlValue]) -> String {
    glsl_with_channels(t, parsed, values, &vec![1; parsed.channel_rank])
}

/// [`glsl`] for an image whose channel dimensions have `extents`.
pub fn glsl_with_channels(t: DataType, parsed: &ParsedShader, values: &[ControlValue], extents: &[u32]) -> String {
    glsl_for(t, parsed, values, extents, false)
}

/// The image layer's emit functions: in a cross-section the colour, its
/// alpha the opacity (`defineImageLayerShader`); in volume rendering
/// (`glsl_emitRGBAVolumeRendering`'s callers) the sample's colour and
/// alpha, composited by the ray march, grey with alpha its value.
fn emit_glsl(volume_rendering: bool) -> &'static str {
    if volume_rendering {
        "#define VOLUME_RENDERING true\n#define uOpacity ng_controls[0]\nvoid emit(vec4 color) { ng_emitted = color; }\nvoid emitRGBA(vec4 rgba) { ng_emitted = rgba; }\nvoid emitRGB(vec3 rgb) { emitRGBA(vec4(rgb, 1.0)); }\nvoid emitGrayscale(float value) { emitRGBA(vec4(value, value, value, value)); }\nvoid emitIntensity(float value) { ng_intensity = value; }\nvoid emitTransparent() { emitIntensity(0.0); emitRGBA(vec4(0.0, 0.0, 0.0, 0.0)); }\n"
    } else {
        "#define VOLUME_RENDERING false\n#define uOpacity ng_controls[0]\nvoid emit(vec4 color) { ng_emitted = color; }\nvoid emitRGBA(vec4 rgba) { emit(vec4(rgba.rgb, rgba.a * uOpacity)); }\nvoid emitRGB(vec3 rgb) { emit(vec4(rgb, uOpacity)); }\nvoid emitGrayscale(float value) { emit(vec4(value, value, value, uOpacity)); }\nvoid emitTransparent() { emit(vec4(0.0, 0.0, 0.0, 0.0)); }\nvoid emitIntensity(float value) { }\n"
    }
}

fn glsl_for(t: DataType, parsed: &ParsedShader, values: &[ControlValue], extents: &[u32], volume_rendering: bool) -> String {
    let n = uniforms(parsed, t, values, 0.0).len();
    let (ty, types) = data_type_glsl(t);
    let mut s = format!(
        r#"#version 450
#define highp
#define mediump
#define lowp
float ng_controls[{n}];
vec4 ng_raw_f;
uvec4 ng_raw_u;
vec4 ng_emitted;
float ng_intensity;
float ng_default_intensity;
{emit}{STRING_T}{types}{access}{COLORMAPS}"#,
        emit = emit_glsl(volume_rendering),
        access = channel_access_glsl(ty, raw_name(t), extents)
    );
    if parsed.controls.iter().any(|(_, c)| matches!(c, Control::TransferFunction { .. })) {
        s += LUT_STUB;
    }
    let mut k = 1;
    for ((name, control), value) in parsed.controls.iter().zip(values) {
        match control {
            Control::Slider { value_type, .. } => {
                let v = format!("ng_controls[{k}]");
                let expr = match value_type {
                    SliderType::Float => v,
                    other => format!("{}({v})", other.glsl()),
                };
                s += &format!("#define {name} {expr}\n");
            }
            Control::Color { .. } => s += &format!("#define {name} vec3(ng_controls[{k}], ng_controls[{}], ng_controls[{}])\n", k + 1, k + 2),
            Control::Checkbox { default } => {
                let on = match value {
                    ControlValue::Bool(b) => *b,
                    _ => *default,
                };
                s += &format!("#define {name} {on}\n");
            }
            Control::Invlerp { clamp, channel, .. } => {
                let channel = match value {
                    ControlValue::Invlerp { channel, .. } => channel,
                    _ => channel,
                };
                s += &invlerp_glsl(t, name, k, *clamp, channel)
            }
            Control::TransferFunction { default } => {
                let base = lookup_table_bases(parsed, t, n)[parsed.controls.iter().take_while(|(m, _)| m != name).filter(|(_, c)| matches!(c, Control::TransferFunction { .. })).count()];
                let channel = match value {
                    ControlValue::TransferFunction(tf) => &tf.channel,
                    _ => &default.channel,
                };
                s += &transfer_function_glsl(t, name, k, base, channel);
            }
            Control::Select { value_type, .. } => {
                let v = format!("ng_controls[{k}]");
                s += &match value_type {
                    SelectType::String => format!("#define {name} string_t(uint({v}))\n"),
                    SelectType::Int => format!("#define {name} int({v})\n"),
                    SelectType::Uint => format!("#define {name} uint({v})\n"),
                    SelectType::Float => format!("#define {name} {v}\n"),
                };
            }
        }
        k += width(control);
    }
    s += "#define main ng_user_main\n";
    s += &parsed.code;
    s += "\n#undef main\nvoid main() { ng_user_main(); }\n";
    s
}

/// What compiling a user shader gives: the sample type and the WGSL of
/// `slice_color` (for `SliceRenderer::with_lookup`, the lookup holding
/// [`uniforms`] as bits), and how many uniforms it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledShader {
    pub sample: VolumeSample,
    pub wgsl: String,
    pub uniforms: usize,
    /// How many texels the uniform texture holds (the uniforms, then
    /// lookup tables).
    pub texels: usize,
}

/// Why a shader did not compile: the line of the user's code (from 0) the
/// compiler points at, if it does, and its message.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileError {
    pub line: Option<usize>,
    pub message: String,
}

/// `parsed` compiled for data of type `t` (checkboxes at `values`): its
/// GLSL translated to WGSL, wrapped as `slice_color`.
pub fn compile(t: DataType, parsed: &ParsedShader, values: &[ControlValue]) -> Result<CompiledShader, CompileError> {
    compile_with_channels(t, parsed, values, &[])
}

/// `parsed` compiled for volume rendering (`VolumeRenderingRenderLayer`'s
/// "on" mode): a WGSL module of `vs_volume` (a chunk's box, its back faces
/// to draw) and `fs_volume` (the ray through the chunk sampled at the
/// depth steps, each sample through the user's shader and composited:
/// `glsl_emitRGBAVolumeRendering`), writing the transparent pass's
/// accumulation and revealage. Bindings: the uniforms, the chunk texture
/// (channels stacked along z), the controls' lookup and the Z buffer.
pub fn compile_volume_rendering(t: DataType, parsed: &ParsedShader, values: &[ControlValue], extents: &[u32]) -> Result<CompiledShader, CompileError> {
    compile_volume_rendering_mode(t, parsed, values, extents, VolumeMode::On)
}

/// `VolumeRenderingModes` other than off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolumeMode {
    /// Composited (`glsl_emitRGBAVolumeRendering`).
    On,
    /// Maximum (or minimum) intensity projection: the sample of the
    /// greatest intensity (`1 −` it for min), its colour and depth; the
    /// intensity as the fragment's depth (kept greatest across chunks).
    Max,
    Min,
}

/// [`compile_volume_rendering`] in `mode`; `fs_volume` returns
/// [`OitOut`]-style targets for [`VolumeMode::On`], else the projection's
/// colour and `1 − depth`, with the intensity as its depth.
pub fn compile_volume_rendering_mode(t: DataType, parsed: &ParsedShader, values: &[ControlValue], extents: &[u32], mode: VolumeMode) -> Result<CompiledShader, CompileError> {
    if let Some(e) = parsed.errors.first() {
        return Err(CompileError { line: Some(e.line), message: e.message.clone() });
    }
    let source = glsl_for(t, parsed, values, extents, true);
    let prelude = source.find(&parsed.code).map_or(0, |at| source[..at].matches('\n').count());
    let error = |location: Option<naga::SourceLocation>, message: String| CompileError {
        line: location.and_then(|l| (l.line_number as usize).checked_sub(1 + prelude)).filter(|&l| l <= parsed.code.matches('\n').count()),
        message,
    };
    let mut frontend = naga::front::glsl::Frontend::default();
    let module = frontend.parse(&naga::front::glsl::Options::from(naga::ShaderStage::Fragment), &source).map_err(|e| {
        let location = e.errors.first().and_then(|first| first.meta.location(&source).into());
        error(location, e.errors.iter().map(|x| x.kind.to_string()).collect::<Vec<_>>().join("\n"))
    })?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| error(e.location(&source), e.as_inner().to_string()))?;
    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).map_err(|e| error(None, e.to_string()))?;
    let wgsl = with_body(&without_entry_point(&wgsl), "fn ng_lut(", "let w = i32(textureDimensions(slice_lookup).x);\n    let j = i + 0;\n    return textureLoad(slice_lookup, vec2<i32>(j % w, j / w), 0).r;");
    let wgsl = with_body(&wgsl, "fn ng_fetch(", "let j = i + 0;\n    return textureLoad(volume, ng_voxel + vec3<i32>(0, 0, j * ng_depth), 0);");
    let n = uniforms(parsed, t, values, 0.0).len();
    let (sample, scalar, raw) = match t {
        DataType::Uint32 | DataType::Int32 | DataType::Uint64 => (VolumeSample::Uint, "u32", "ng_raw_u"),
        _ => (VolumeSample::Float, "f32", "ng_raw_f"),
    };
    let channels = extents.iter().map(|&e| e.max(1)).product::<u32>();
    let module = format!(
        r#"{oit}
struct VolumeUniforms {{
    mvp : mat4x4<f32>,
    inv_mvp : mat4x4<f32>,
    translation : vec4<f32>,
    chunk_size : vec4<f32>,
    lower_clip : vec4<f32>,
    upper_clip : vec4<f32>,
    // Near and far limit fractions, brightness factor, gain.
    params : vec4<f32>,
    // Maximum steps.
    steps : vec4<u32>,
}};
@group(0) @binding(0) var<uniform> vr : VolumeUniforms;
@group(0) @binding(1) var volume : texture_3d<{scalar}>;
@group(0) @binding(2) var slice_lookup : texture_2d<u32>;
@group(0) @binding(3) var vr_depth : texture_2d<f32>;
var<private> ng_voxel : vec3<i32>;
var<private> ng_depth : i32;
{wgsl}
struct VolumeOut {{
    @builtin(position) clip : vec4<f32>,
    @location(0) ndc : vec4<f32>,
}};
// `glsl_getBoxFaceVertexPosition`: the unit cube's faces, outward
// counter-clockwise.
fn ng_box_vertex(i : u32) -> vec3<f32> {{
    var corners = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var indices = array<u32, 36>(
        0u, 2u, 1u, 1u, 2u, 3u,
        4u, 5u, 6u, 5u, 7u, 6u,
        0u, 1u, 4u, 1u, 5u, 4u,
        2u, 6u, 3u, 3u, 6u, 7u,
        0u, 4u, 2u, 2u, 4u, 6u,
        1u, 3u, 5u, 3u, 7u, 5u);
    return corners[indices[i]];
}}
@vertex
fn vs_volume(@builtin(vertex_index) i : u32) -> VolumeOut {{
    var out : VolumeOut;
    let p = max(vr.lower_clip.xyz, min(vr.upper_clip.xyz, vr.translation.xyz + ng_box_vertex(i) * vr.chunk_size.xyz));
    let c = vr.mvp * vec4<f32>(p, 1.0);
    out.ndc = c;
    // Mirrored for the upside-down target; depth 0 (no clipping by it).
    out.clip = vec4<f32>(c.x, -c.y, 0.0, c.w);
    return out;
}}
{projection_out}
@fragment
fn fs_volume(in : VolumeOut) -> {fs_out} {{
    let w = i32(textureDimensions(slice_lookup).x);
    for (var i = 0; i < {n}; i++) {{
        ng_controls[i] = bitcast<f32>(textureLoad(slice_lookup, vec2<i32>(i % w, i / w), 0).r);
    }}
    let ndc = in.ndc.xy / in.ndc.w;
    let near_h = vr.inv_mvp * vec4<f32>(ndc, -1.0, 1.0);
    let far_h = vr.inv_mvp * vec4<f32>(ndc, 1.0, 1.0);
    let near_point = near_h.xyz / near_h.w;
    let far_point = far_h.xyz / far_h.w;
    let ray = far_point - near_point;
    let box_start = max(vr.lower_clip.xyz, vr.translation.xyz);
    let box_end = min(box_start + vr.chunk_size.xyz, vr.upper_clip.xyz);
    let near_fraction = vr.params.x;
    let far_fraction = vr.params.y;
    var intersect_start = near_fraction;
    var intersect_end = far_fraction;
    for (var k = 0; k < 3; k++) {{
        let start_pt = near_point[k];
        let end_pt = far_point[k];
        let lower = box_start[k];
        let upper = box_end[k];
        let r = ray[k];
        var start_f : f32;
        var end_f : f32;
        if (start_pt >= lower && start_pt <= upper) {{
            start_f = 0.0;
        }} else {{
            start_f = min((lower - start_pt) / r, (upper - start_pt) / r);
        }}
        if (end_pt >= lower && end_pt <= upper) {{
            end_f = 1.0;
        }} else {{
            end_f = max((lower - start_pt) / r, (upper - start_pt) / r);
        }}
        intersect_start = max(intersect_start, start_f);
        intersect_end = min(intersect_end, end_f);
    }}
    let max_steps = i32(vr.steps.x);
    let step_size = (far_fraction - near_fraction) / f32(max_steps - 1);
    let start_step = i32(floor((intersect_start - near_fraction) / step_size));
    let end_step = min(max_steps, i32(floor((intersect_end - near_fraction) / step_size)) + 1);
    var output_color = vec4<f32>(0.0);
    var revealage = 1.0;
    var saved_intensity = 0.0;
    var saved_depth = 0.0;
    let clip_near = vr.mvp * vec4<f32>(near_point, 1.0);
    let clip_far = vr.mvp * vec4<f32>(far_point, 1.0);
    let depth_in_buffer = textureLoad(vr_depth, vec2<i32>(in.clip.xy), 0).r;
    let size = vec3<i32>(textureDimensions(volume));
    ng_depth = size.z / {channels};
    let extent = vec3<i32>(size.x, size.y, ng_depth);
    for (var ray_step = start_step; ray_step < end_step; ray_step++) {{
        let fraction = near_fraction + f32(ray_step) * step_size;
        let position = mix(near_point, far_point, fraction);
        let clip_position = mix(clip_near, clip_far, fraction);
        let depth = (clip_position.z / clip_position.w + 1.0) * 0.5;
        // Behind an opaque object.
        if ((1.0 - depth) < depth_in_buffer) {{
            break;
        }}
        let in_chunk = position - vr.translation.xyz;
        ng_voxel = clamp(vec3<i32>(floor(in_chunk)), vec3<i32>(0), extent - vec3<i32>(1));
        {raw} = textureLoad(volume, ng_voxel, 0);
{step_body}
    }}
{fs_return}
}}
"#,
        oit = crate::perspective::OIT_SHADER,
        projection_out = if mode == VolumeMode::On { "" } else { "struct ProjectionOut {\n    @location(0) color : vec4<f32>,\n    @location(1) z : vec4<f32>,\n    @builtin(frag_depth) depth : f32,\n};\nvar<private> ng_new_color : vec4<f32>;" },
        fs_out = if mode == VolumeMode::On { "OitOut" } else { "ProjectionOut" },
        step_body = match mode {
            VolumeMode::On => "        ng_emitted = vec4<f32>(0.0);\n        ng_user_main();\n        let corrected = clamp(ng_emitted.a * vr.params.z * vr.params.w, 0.0, 1.0);\n        let weighted = corrected * compute_oit_weight(corrected, depth);\n        output_color += vec4<f32>(ng_emitted.rgb * weighted, weighted);\n        revealage *= 1.0 - corrected;".to_owned(),
            // The sample's colour stays until another `emitRGBA`; the
            // intensity is the one emitted, else the invlerp's.
            _ => format!(
                "        ng_emitted = vec4<f32>(-1.0);\n        ng_intensity = -100.0;\n        ng_default_intensity = 0.0;\n        ng_user_main();\n        if (ng_emitted.a >= 0.0) {{\n            let alpha = clamp(ng_emitted.a, 0.0, 1.0);\n            ng_new_color = vec4<f32>(ng_emitted.rgb * alpha, alpha);\n        }}\n        let emitted = select(ng_default_intensity, ng_intensity, ng_intensity > -100.0);\n        let intensity = clamp({conv}, 0.0, 1.0);\n        if (intensity > saved_intensity) {{\n            saved_intensity = intensity;\n            saved_depth = depth;\n            output_color = ng_new_color;\n        }}",
                conv = if mode == VolumeMode::Min { "1.0 - emitted" } else { "emitted" }
            ),
        },
        fs_return = if mode == VolumeMode::On {
            "    return OitOut(vec4<f32>(output_color.rgb, 1.0 - revealage), vec4<f32>(output_color.a, 0.0, 0.0, 0.0));"
        } else {
            "    let buffer_depth = 1.0 - saved_depth;\n    return ProjectionOut(output_color, vec4<f32>(buffer_depth, buffer_depth, buffer_depth, 1.0), saved_intensity);"
        },
    );
    let texels = uniform_bits(parsed, t, values, 0.0).len();
    Ok(CompiledShader { sample, wgsl: module, uniforms: n, texels })
}

/// The variable `ng_value` reads: integer types of 32 bits and more come
/// from a `u32` texture.
fn raw_name(t: DataType) -> &'static str {
    match t {
        DataType::Uint32 | DataType::Int32 | DataType::Uint64 => "ng_raw_u",
        _ => "ng_raw_f",
    }
}

/// [`compile`] for an image with channel dimensions of `extents` (first
/// fastest), their channels stacked along the chunk texture's z
/// (`slice_color_at`: each channel's slab is the texture's depth over
/// their count).
pub fn compile_with_channels(t: DataType, parsed: &ParsedShader, values: &[ControlValue], extents: &[u32]) -> Result<CompiledShader, CompileError> {
    if let Some(e) = parsed.errors.first() {
        return Err(CompileError { line: Some(e.line), message: e.message.clone() });
    }
    let source = glsl_with_channels(t, parsed, values, extents);
    // The user's code starts after the helpers.
    let prelude = source.find(&parsed.code).map_or(0, |at| source[..at].matches('\n').count());
    let error = |location: Option<naga::SourceLocation>, message: String| CompileError {
        line: location.and_then(|l| (l.line_number as usize).checked_sub(1 + prelude)).filter(|&l| l <= parsed.code.matches('\n').count()),
        message,
    };
    let mut frontend = naga::front::glsl::Frontend::default();
    let module = frontend.parse(&naga::front::glsl::Options::from(naga::ShaderStage::Fragment), &source).map_err(|e| {
        let location = e.errors.first().and_then(|first| first.meta.location(&source).into());
        error(location, e.errors.iter().map(|x| x.kind.to_string()).collect::<Vec<_>>().join("\n"))
    })?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| error(e.location(&source), e.as_inner().to_string()))?;
    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).map_err(|e| error(None, e.to_string()))?;
    let wgsl = with_body(&without_entry_point(&wgsl), "fn ng_lut(", "let w = i32(textureDimensions(slice_lookup).x);\n    let j = i + 0;\n    return textureLoad(slice_lookup, vec2<i32>(j % w, j / w), 0).r;");
    let wgsl = if extents.is_empty() { wgsl } else { with_body(&wgsl, "fn ng_fetch(", "let j = i + 0;\n    return textureLoad(volume, ng_voxel + vec3<i32>(0, 0, j * ng_depth), 0);") };
    let n = uniforms(parsed, t, values, 0.0).len();
    let (sample, scalar, raw) = match t {
        DataType::Uint32 | DataType::Int32 | DataType::Uint64 => (VolumeSample::Uint, "u32", "ng_raw_u"),
        _ => (VolumeSample::Float, "f32", "ng_raw_f"),
    };
    let wrapper = format!(
        r#"
fn slice_color(value : vec4<{scalar}>, params : array<vec4<f32>, 4>) -> vec4<f32> {{
    let w = i32(textureDimensions(slice_lookup).x);
    for (var i = 0; i < {n}; i++) {{
        ng_controls[i] = bitcast<f32>(textureLoad(slice_lookup, vec2<i32>(i % w, i / w), 0).r);
    }}
    {raw} = value;
    ng_emitted = vec4<f32>(0.0);
    ng_user_main();
    return ng_emitted;
}}
"#
    );
    // Channels: the voxel within the first channel's slab, kept for the
    // others' reads.
    let channels = extents.iter().map(|&e| e.max(1)).product::<u32>();
    let at = if extents.is_empty() {
        String::new()
    } else {
        format!(
            r#"
var<private> ng_voxel : vec3<i32>;
var<private> ng_depth : i32;
fn slice_color_at(voxel : vec3<f32>, params : array<vec4<f32>, 4>) -> vec4<f32> {{
    let size = vec3<i32>(textureDimensions(volume));
    ng_depth = size.z / {channels};
    let v = vec3<i32>(voxel);
    if (any(voxel < vec3<f32>(0.0)) || v.x >= size.x || v.y >= size.y || v.z >= ng_depth) {{
        discard;
    }}
    ng_voxel = v;
    return slice_color(textureLoad(volume, v, 0), params);
}}
"#
        )
    };
    let texels = uniform_bits(parsed, t, values, 0.0).len();
    Ok(CompiledShader { sample, wgsl: format!("{wgsl}{wrapper}{at}"), uniforms: n, texels })
}

/// `wgsl` with the body of the function starting `head` (naga's
/// `fn name(`) replaced, its parameter (one, `i`) renamed as naga named it.
fn with_body(wgsl: &str, head: &str, body: &str) -> String {
    let Some(start) = wgsl.find(head) else { return wgsl.to_owned() };
    let params_end = wgsl[start..].find(')').map_or(start, |e| start + e);
    let param = wgsl[start + head.len()..params_end].split(':').next().unwrap_or("i").trim().to_owned();
    let Some(open) = wgsl[start..].find('{').map(|o| start + o) else { return wgsl.to_owned() };
    let mut depth = 0;
    for (i, c) in wgsl[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = open + i;
                    let body = body.replace("let j = i + 0;", &format!("let j = {param};"));
                    return format!("{}{{\n    {body}\n{}", &wgsl[..open], &wgsl[end..]);
                }
            }
            _ => {}
        }
    }
    wgsl.to_owned()
}

/// naga's WGSL without the `@fragment` function (the pipeline has its
/// own).
pub(crate) fn without_entry_point(wgsl: &str) -> String {
    let Some(start) = wgsl.find("@fragment") else { return wgsl.to_owned() };
    let Some(open) = wgsl[start..].find('{').map(|o| start + o) else { return wgsl.to_owned() };
    let mut depth = 0;
    for (i, c) in wgsl[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = open + i + 1;
                    return format!("{}{}", &wgsl[..start], &wgsl[end..]);
                }
            }
            _ => {}
        }
    }
    wgsl.to_owned()
}

/// The uniforms' texture (the slice pipeline's lookup): each value's bits,
/// in an `R32Uint` row.
pub struct UniformTexture {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    len: usize,
}

impl UniformTexture {
    /// Rows this wide at most (WebGL2's guaranteed texture size).
    const WIDTH: usize = 2048;

    pub fn new(device: &wgpu::Device, len: usize) -> Self {
        let len = len.max(1);
        let width = len.min(Self::WIDTH) as u32;
        let height = len.div_ceil(Self::WIDTH) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ng shader controls"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view, len: (width * height) as usize }
    }

    /// Writes `bits` ([`uniform_bits`]; as many as the texture holds), row
    /// by row.
    pub fn set(&self, queue: &wgpu::Queue, bits: &[u32]) {
        let mut bits = bits.to_vec();
        bits.resize(self.len, 0);
        let width = self.len.min(Self::WIDTH) as u32;
        let height = self.len as u32 / width;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&bits),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * width), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_directives_as_neuroglancer() {
        let code = "#uicontrol invlerp normalized\n// a comment #uicontrol bool x\n#uicontrol float brightness slider(min=-1, max=1, step=0.1)\n#uicontrol vec3 color color(default=\"#f00\")\n#uicontrol bool on checkbox(default=true)\nvoid main() {\n  emitRGB(color * (normalized() + brightness));\n}\n";
        let p = parse(code, DataType::Uint8);
        assert!(p.errors.is_empty(), "{:?}", p.errors);
        let names: Vec<&str> = p.controls.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["normalized", "brightness", "color", "on"]);
        assert_eq!(p.controls[1].1, Control::Slider { value_type: SliderType::Float, min: -1.0, max: 1.0, step: 0.1, default: 0.0 });
        assert_eq!(p.controls[2].1, Control::Color { default: [1.0, 0.0, 0.0], default_string: "#f00".into() });
        assert_eq!(p.code.lines().count(), code.lines().count());
        assert!(!p.code.contains("uicontrol"));
        let bad = parse("#uicontrol float x slider(max=1)\n#uicontrol int y slider(min=0, max=1, step=0.5)\n#uicontrol q\n", DataType::Uint8);
        let messages: Vec<(usize, &str)> = bad.errors.iter().map(|e| (e.line, e.message.as_str())).collect();
        assert_eq!(messages, [(0, "min must be specified"), (1, "Expected step argument to be an integer"), (2, "Invalid #uicontrol syntax, expected: #uicontrol <type> <name> <control>(<param>=<value>, ...)")]);
        let (params, errors) = parse_directive_parameters(Some("range=[0, 100], clamp=false"));
        assert!(errors.is_empty());
        assert_eq!(params, vec![("range".into(), json!([0, 100])), ("clamp".into(), json!(false))]);
    }

    #[test]
    fn channels_are_read_by_index() {
        // `channel` is checked against the image's channel rank.
        let parsed = parse_with_channels("#uicontrol invlerp normalized(channel=[2])\nvoid main() { emitGrayscale(normalized()); }\n", DataType::Uint8, 1);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        assert!(matches!(&parsed.controls[0].1, Control::Invlerp { channel, .. } if channel == &vec![2]));
        let bad = parse_with_channels("#uicontrol invlerp normalized(channel=[1, 2])\nvoid main() {}\n", DataType::Uint8, 1);
        assert_eq!(bad.errors[0].message, "Invalid channel value: Expected array of length 1, but received: [1,2]");
        assert!(!parse("#uicontrol invlerp normalized(channel=0)\nvoid main() {}\n", DataType::Uint8).errors.is_empty());
        // The state's channel overrides the directive's, compiled in.
        let values = control_values(&parsed, DataType::Uint8, Some(&serde_json::json!({ "normalized": { "channel": [1] } })));
        assert!(matches!(&values[0], ControlValue::Invlerp { channel, .. } if channel == &vec![1]));
        assert!(glsl_with_channels(DataType::Uint8, &parsed, &values, &[3]).contains("getDataValue(1)"));
        for t in DataType::ALL {
            let parsed = parse_with_channels("void main() { getDataValue(0); getDataValue(1); emitGrayscale(0.5); }\n", t, 1);
            let compiled = compile_with_channels(t, &parsed, &[], &[3]).unwrap_or_else(|e| panic!("{t:?}: {}", e.message));
            assert!(compiled.wgsl.contains("fn slice_color_at(") && compiled.wgsl.contains("ng_depth = size.z / 3"), "{t:?}");
            assert!(compiled.wgsl.contains("textureLoad(volume, ng_voxel + vec3<i32>(0, 0, "), "{t:?}");
            // Valid WGSL, with the slice pipeline's bindings around it.
            let sample = if matches!(compiled.sample, VolumeSample::Uint) { "u32" } else { "f32" };
            let full = format!("@group(0) @binding(1) var volume : texture_3d<{sample}>;\n@group(0) @binding(2) var slice_lookup : texture_2d<u32>;\n{}", compiled.wgsl);
            naga::front::wgsl::parse_str(&full).unwrap_or_else(|e| panic!("{t:?}: {}", e.emit_to_string(&full)));
        }
    }

    #[test]
    fn compiles_for_volume_rendering() {
        for t in DataType::ALL {
            for (extents, mode) in [(vec![], VolumeMode::On), (vec![3], VolumeMode::On), (vec![], VolumeMode::Max), (vec![3], VolumeMode::Min)] {
                let parsed = parse_with_channels(crate::user_shader::tests::DEFAULT, t, extents.len());
                let values = control_values(&parsed, t, None);
                let compiled = compile_volume_rendering_mode(t, &parsed, &values, &extents, mode).unwrap_or_else(|e| panic!("{t:?}: {}", e.message));
                let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap_or_else(|e| panic!("{t:?}: {}", e.emit_to_string(&compiled.wgsl)));
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{t:?}: {e:?}"));
            }
        }
    }

    const DEFAULT: &str = "#uicontrol invlerp normalized\nvoid main() {\n  emitGrayscale(normalized());\n}\n";

    #[test]
    fn compiles_to_wgsl_for_every_data_type() {
        let code = "#uicontrol invlerp normalized(range=[10, 20])\n#uicontrol float gain slider(min=0, max=2)\n#uicontrol vec3 tint color\n#uicontrol bool jet checkbox\nvoid main() {\n  float v = normalized() * gain;\n  if (jet) { emitRGB(colormapJet(v)); } else { emitRGBA(vec4(tint * v, 0.5)); }\n}\n";
        for t in [DataType::Uint8, DataType::Int8, DataType::Uint16, DataType::Int16, DataType::Uint32, DataType::Int32, DataType::Uint64, DataType::Float32] {
            let p = parse(code, t);
            let values = control_values(&p, t, Some(&json!({ "gain": 1.5, "jet": true })));
            let compiled = compile(t, &p, &values).unwrap_or_else(|e| panic!("{t:?}: {e:?}"));
            assert!(compiled.wgsl.contains("fn slice_color"));
            assert!(!compiled.wgsl.contains("@fragment"));
            let u = uniforms(&p, t, &values, 0.25);
            assert_eq!(u.len(), compiled.uniforms);
            assert_eq!((u[0], u[9]), (0.25, 1.5));
            // The WGSL, with the pipeline's declarations, validates.
            let lookup = "@group(0) @binding(2) var slice_lookup : texture_2d<u32>;\n";
            let module = naga::front::wgsl::parse_str(&format!("{lookup}{}", compiled.wgsl)).unwrap_or_else(|e| panic!("{t:?}: {}", e.emit_to_string(&compiled.wgsl)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        }
        // A compile error carries the compiler's message.
        let p = parse("#uicontrol float g slider(min=0, max=1)\nvoid main() {\n  emitRGB(nope);\n}\n", DataType::Uint8);
        let e = compile(DataType::Uint8, &p, &control_values(&p, DataType::Uint8, None)).unwrap_err();
        assert_eq!(e.line, Some(2), "{e:?}");
        assert!(e.message.contains("nope"), "{e:?}");
        let p = parse("#uicontrol float g slider(max=1)\nvoid main() {}\n", DataType::Uint8);
        let e = compile(DataType::Uint8, &p, &control_values(&p, DataType::Uint8, None)).unwrap_err();
        assert_eq!((e.line, e.message.as_str()), (Some(0), "min must be specified"));
    }

    #[test]
    fn select_controls_as_neuroglancer() {
        let t = DataType::Uint8;
        let code = "#uicontrol string_t mode select(options={\"Grayscale\": \"gray\", \"Jet\": \"jet\"}, default=\"gray\")\n#uicontrol int k select(options=[1, 2, 3])\n#uicontrol float f select(options={\"half\": 0.5, \"one\": 1})\nvoid main() {\n  float v = toNormalized(getDataValue()) * f * float(k);\n  if (mode == \"jet\") { emitRGB(colormapJet(v)); } else { emitGrayscale(v); }\n}\n";
        let p = parse(code, t);
        assert!(p.errors.is_empty(), "{:?}", p.errors);
        assert_eq!(p.string_literals, vec!["jet".to_owned()]);
        assert!(p.code.contains("mode == string_t(1u)"));
        let Control::Select { value_type, options, default } = &p.controls[0].1 else { panic!() };
        assert_eq!((*value_type, default), (SelectType::String, &SelectValue::String("gray".into())));
        assert_eq!(options.iter().map(SelectOption::display).collect::<Vec<_>>(), vec!["Grayscale (gray)", "Jet (jet)"]);
        let Control::Select { default, options, .. } = &p.controls[1].1 else { panic!() };
        assert_eq!((default, options[2].display().as_str()), (&SelectValue::Number(1.0), "3"));
        // "gray" is not in the code: id 0; "jet" is 1.
        let values = control_values(&p, t, None);
        assert_eq!(uniforms(&p, t, &values, 1.0), vec![1.0, 0.0, 1.0, 0.5]);
        let json = serde_json::json!({"mode": "jet", "k": 3, "f": 7});
        let values = control_values(&p, t, Some(&json));
        assert_eq!(uniforms(&p, t, &values, 1.0), vec![1.0, 1.0, 3.0, 0.5]);
        compile(t, &p, &values).unwrap();
        let errors = |code: &str| parse(code, t).errors.into_iter().map(|e| e.message).collect::<Vec<_>>();
        assert_eq!(errors("#uicontrol vec3 m select(options=[1])"), vec!["type must be one of string_t, int, uint, or float"]);
        assert_eq!(errors("#uicontrol int m select(default=1)"), vec!["options must be specified"]);
        assert_eq!(errors("#uicontrol int m select(options=[1, 1])"), vec!["Duplicate option value: 1"]);
        assert_eq!(errors("#uicontrol int m select(options=[])"), vec!["Expected options argument to contain at least one option"]);
        assert_eq!(errors("#uicontrol uint m select(options=[1], default=2)"), vec!["default value 2 must match one of the options"]);
        assert_eq!(errors("#uicontrol string_t m select(options=1)"), vec!["Expected options argument to be an array of string_t values or an object mapping labels to string_t values"]);
    }

    #[test]
    fn transfer_functions_compile_and_pack_their_tables() {
        let code = "#uicontrol transferFunction colormap(controlPoints=[[0, \"#000000\", 0], [255, \"#ffffff\", 1]])\nvoid main() {\n  emitRGBA(colormap());\n}\n";
        for t in [DataType::Uint8, DataType::Int8, DataType::Uint16, DataType::Int16, DataType::Uint32, DataType::Int32, DataType::Uint64, DataType::Float32] {
            let p = parse(&code.replace("255", "100"), t);
            assert!(p.errors.is_empty(), "{t:?}: {:?}", p.errors);
            let values = control_values(&p, t, None);
            let compiled = compile(t, &p, &values).unwrap_or_else(|e| panic!("{t:?}: {e:?}"));
            assert!(compiled.wgsl.contains("textureLoad(slice_lookup"), "{}", compiled.wgsl);
            assert!(!compiled.wgsl.contains("3735928559"), "the stub replaced");
            assert_eq!(compiled.texels, 9 + crate::transfer_function::lookup_table_size(t));
            let bits = uniform_bits(&p, t, &values, 1.0);
            assert_eq!(bits.len(), compiled.texels);
        }
        let t = DataType::Uint8;
        let p = parse(code, t);
        let bits = uniform_bits(&p, t, &control_values(&p, t, None), 1.0);
        assert_eq!(bits[9], 0);
        assert_eq!(bits[9 + 255], u32::MAX);
        // The window defaults to the points' range; the state's points apply.
        let Control::TransferFunction { default } = &p.controls[0].1 else { panic!() };
        assert_eq!(default.window, Interval::Number([0.0, 255.0]));
        let json = serde_json::json!({"colormap": {"controlPoints": [[100, "#ff0000", 1]]}});
        let values = control_values(&p, t, Some(&json));
        let ControlValue::TransferFunction(tf) = &values[0] else { panic!() };
        assert_eq!(tf.points.range, Interval::Number([100.0, 255.0]));
        assert_eq!(tf.to_json(default), Some(serde_json::json!({"controlPoints": [[100, "#ff0000", 1]]})));
        let errors = |code: &str| parse(code, t).errors.into_iter().map(|e| e.message).collect::<Vec<_>>();
        assert_eq!(errors("#uicontrol transferFunction c(foo=1)"), vec!["Invalid parameter: foo"]);
        assert!(errors("#uicontrol transferFunction c(controlPoints=[[0, \"#fff\", 1]])")[0].starts_with("Invalid controlPoints value: Expected #RRGGBB"));
    }
}
