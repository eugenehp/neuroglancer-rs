//! Viewer state JSON: the parts of neuroglancer's state this viewer acts on,
//! read as neuroglancer reads them (`src/viewer.ts`, `src/layer/index.ts`,
//! `src/layer/layer_data_source.ts`, the image and segmentation layers).
//! Unknown keys are kept on each layer, so they can be written back.

use anyhow::{anyhow, bail, Context, Result};
use ng_core::transform::{legacy_transform_spec, TransformSpec};
use ng_core::{js, DataType};
use ng_render::image::Interval;
use ng_render::segmentation::SegmentationDisplay;
use serde_json::{Map, Value};

/// The viewer state's keys in the order neuroglancer writes them
/// (`ViewerState`'s `add` calls, `src/viewer.ts`).
pub const VIEWER_STATE_KEYS: &[&str] = &[
    "title", "dimensions", "relativeDisplayScales", "displayDimensions", "position", "velocity",
    "crossSectionOrientation", "crossSectionScale", "crossSectionDepth", "projectionOrientation",
    "projectionScale", "projectionDepth", "layers", "showAxisLines", "wireFrame", "enableAdaptiveDownsampling",
    "showScaleBar", "showDefaultAnnotations", "showSlices", "hideCrossSectionBackground3D", "gpuMemoryLimit",
    "prefetch", "systemMemoryLimit", "concurrentDownloads", "selectedLayer", "crossSectionBackgroundColor",
    "projectionBackgroundColor", "layout", "statistics", "helpPanel", "settingsPanel", "selection",
    "layerListPanel", "partialViewport", "selectedStateServer", "toolBindings", "toolPalettes",
    "uiControlVisibility",
];

/// `serializeColor` of an RGB colour: `#rrggbb`.
pub fn serialize_color(c: [f32; 3]) -> String {
    let byte = |x: f32| (js::round(x as f64 * 255.0)).clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", byte(c[0]), byte(c[1]), byte(c[2]))
}

/// One data source of a layer (`DataSourceSpecification`).
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSpec {
    pub url: String,
    pub transform: Option<TransformSpec>,
    /// `enableDefaultSubsources` (default true).
    pub enable_default_subsources: bool,
    /// `subsources`: explicitly enabled or disabled ones.
    pub subsources: std::collections::BTreeMap<String, bool>,
}

/// One layer as the state describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    pub name: String,
    /// `type`: `"image"`, `"segmentation"`, `"auto"` (from the data), or one
    /// this viewer does not render yet.
    pub layer_type: String,
    pub visible: bool,
    pub archived: bool,
    pub sources: Vec<SourceSpec>,
    /// The whole layer object, for settings read once the data is known.
    pub json: Map<String, Value>,
}

/// `layerDataSourceSpecificationFromJson`.
fn source_spec(obj: &Value, legacy: Option<&TransformSpec>) -> Result<SourceSpec> {
    if let Some(url) = obj.as_str() {
        return Ok(SourceSpec {
            url: url.to_owned(),
            transform: legacy.cloned(),
            enable_default_subsources: true,
            subsources: Default::default(),
        });
    }
    let o = obj.as_object().ok_or_else(|| anyhow!("Expected object, but received: {obj}"))?;
    let url = o
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Error parsing \"url\" property: Expected string"))?;
    let transform = TransformSpec::from_json(o.get("transform")).context("Error parsing \"transform\" property")?;
    let enable_default_subsources = match o.get("enableDefaultSubsources") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(v) => bail!("Error parsing \"enableDefaultSubsources\" property: Expected boolean, but received: {v}"),
    };
    // `parseDataSubsourceSpecificationFromJson`: a boolean per subsource.
    let mut subsources = std::collections::BTreeMap::new();
    if let Some(v) = o.get("subsources") {
        let m = v.as_object().ok_or_else(|| anyhow!("Error parsing \"subsources\" property: Expected object, but received: {v}"))?;
        for (id, e) in m {
            let enabled = e.as_bool().ok_or_else(|| anyhow!("Error parsing \"subsources\" property: Expected boolean, but received: {e}"))?;
            subsources.insert(id.clone(), enabled);
        }
    }
    Ok(SourceSpec {
        url: url.to_owned(),
        transform: transform.or_else(|| legacy.cloned()),
        enable_default_subsources,
        subsources,
    })
}

/// `getDataSourceSpecifications`: `source` as an array, an object, or a URL
/// string (with the legacy top-level `transform`).
fn source_specs(layer: &Map<String, Value>) -> Result<Vec<SourceSpec>> {
    match layer.get("source") {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Array(a)) => a.iter().map(|s| source_spec(s, None)).collect(),
        Some(o @ Value::Object(_)) => Ok(vec![source_spec(o, None)?]),
        Some(s) => {
            let legacy = legacy_transform_spec(layer.get("transform")).context("Error parsing \"transform\" property")?;
            Ok(vec![source_spec(s, legacy.as_ref())?])
        }
    }
}

fn optional_bool(o: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match o.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => bail!("Error parsing {key:?} property: Expected boolean, but received: {v}"),
    }
}

/// `LayerListSpecification.restoreState`: an array of layers, or the legacy
/// object keyed by name. Names are made unique as `getUniqueLayerName` does.
/// Layers that fail to parse are reported and left out, as neuroglancer does.
pub fn parse_layers(x: &Value) -> Result<(Vec<LayerSpec>, Vec<String>)> {
    let specs: Vec<Value> = match x {
        Value::Array(a) => a.clone(),
        Value::Object(o) => o
            .iter()
            .map(|(name, spec)| match spec {
                Value::String(s) => Ok(serde_json::json!({ "name": name, "source": s })),
                Value::Object(m) => {
                    let mut m = m.clone();
                    m.insert("name".into(), Value::String(name.clone()));
                    Ok(Value::Object(m))
                }
                v => Err(anyhow!("Expected object, but received: {v}")),
            })
            .collect::<Result<_>>()?,
        v => bail!("Expected object, but received: {v}"),
    };
    let mut layers: Vec<LayerSpec> = vec![];
    let mut errors = vec![];
    for spec in specs {
        let o = spec.as_object().ok_or_else(|| anyhow!("Expected object, but received: {spec}"))?;
        let name = o
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Error parsing \"name\" property: Expected string"))?;
        let mut unique = name.to_owned();
        let mut suffix = 0;
        while layers.iter().any(|l| l.name == unique) {
            suffix += 1;
            unique = format!("{name}{suffix}");
        }
        let parsed = (|| -> Result<LayerSpec> {
            let layer_type = match o.get("type") {
                None => "auto".to_owned(),
                Some(Value::String(t)) => t.clone(),
                Some(v) => bail!("Error parsing \"type\" property: Expected string, but received: {v}"),
            };
            let archived = optional_bool(o, "archived", false)?;
            let visible = if archived { false } else { optional_bool(o, "visible", true)? };
            Ok(LayerSpec {
                name: unique.clone(),
                layer_type,
                visible,
                archived,
                sources: source_specs(o)?,
                json: o.clone(),
            })
        })();
        match parsed {
            Ok(l) => layers.push(l),
            Err(e) => errors.push(format!("Error creating layer {unique:?}: {e:#}")),
        }
    }
    Ok((layers, errors))
}

/// `verifyFloat01` through a `TrackableValue`: invalid values give the
/// default.
fn alpha(o: &Map<String, Value>, key: &str, default: f32) -> f32 {
    match o.get(key).and_then(Value::as_f64) {
        Some(x) if x.is_finite() && (0.0..=1.0).contains(&x) => x as f32,
        _ => default,
    }
}

fn boolean(o: &Map<String, Value>, key: &str, default: bool) -> bool {
    match o.get(key) {
        Some(Value::Bool(b)) => *b,
        _ => default,
    }
}

/// `parseDataTypeValue`.
pub fn parse_data_type_value(t: DataType, x: &Value) -> Result<f64> {
    let s = match x {
        Value::String(s) => s.clone(),
        v => js::to_string(v),
    };
    match t {
        DataType::Float32 => {
            let v = js::parse_float(&s);
            if !v.is_finite() {
                bail!("Invalid float32 value: {s:?}");
            }
            Ok(v)
        }
        _ => {
            let v = js::parse_int(&s);
            let [lo, hi] = ng_render::image::default_data_type_range(t).as_f64();
            if v.is_nan() || v.fract() != 0.0 || v < lo || v > hi {
                bail!("Invalid {} value: {s:?}", t.name());
            }
            Ok(v)
        }
    }
}

/// `parseUint64` of `String(x)` (numbers go through JavaScript's string
/// form, so past 2^53 they are what JavaScript made of them).
pub fn parse_uint64(x: &Value) -> Result<u64> {
    let s = match x {
        Value::String(s) => s.clone(),
        Value::Number(n) => js::to_string(&Value::from(n.as_f64().unwrap_or(f64::NAN))),
        v => bail!("Expected uint64 value, but received: {v}"),
    };
    if s != "0" && !(s.starts_with(|c: char| ('1'..='9').contains(&c)) && s.bytes().all(|b| b.is_ascii_digit())) {
        bail!("Expected base-10 number, but received: {s:?}");
    }
    s.parse::<u64>().map_err(|_| anyhow!("Expected uint64 value, but received: {s}"))
}

/// The default image shader (`getDefaultFragmentMain`'s), the only one run so
/// far.
/// `DEFAULT_FRAGMENT_MAIN` of annotation layers.
/// `skeleton/frontend.ts` `DEFAULT_FRAGMENT_MAIN`.
pub const DEFAULT_SKELETON_SHADER: &str = "void main() {\n  emitDefault();\n}\n";

pub const DEFAULT_ANNOTATION_SHADER: &str = "\nvoid main() {\n  setColor(defaultColor());\n}\n";

pub const DEFAULT_IMAGE_SHADER: &str = "#uicontrol invlerp normalized\nvoid main() {\n  emitGrayscale(normalized());\n}\n";

/// An image layer's settings: `opacity`, and the `normalized` control's
/// `range`. A shader other than the default is reported (not run yet).
/// An image layer's volume rendering (`volumeRendering`: `"on"`, `"max"`,
/// `"min"`, `"off"`, or a boolean), `volumeRenderingDepthSamples` (a render
/// scale target, 64) and `volumeRenderingGain` (finite, 0).
pub fn volume_rendering_settings(o: &Map<String, Value>) -> (&'static str, f64, f64) {
    let mode = match o.get("volumeRendering") {
        Some(Value::Bool(true)) => "on",
        Some(Value::String(m)) => match m.to_ascii_lowercase().as_str() {
            "on" => "on",
            "max" => "max",
            "min" => "min",
            _ => "off",
        },
        _ => "off",
    };
    // `trackableRenderScaleTarget(64, 2¹, 2²¹ − 1)`.
    let samples = o.get("volumeRenderingDepthSamples").and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= 2.0 && *v <= 2097151.0).unwrap_or(64.0);
    let gain = o.get("volumeRenderingGain").and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0);
    (mode, samples, gain)
}

pub fn image_settings(o: &Map<String, Value>, t: DataType) -> (f32, Interval, Option<String>) {
    let opacity = alpha(o, "opacity", ng_render::image::DEFAULT_OPACITY);
    // Another shader draws on its own; its fallback (the default, when it
    // does not compile) uses the default's control values
    // (`getFallbackBuilderState`).
    let custom = o.get("shader").and_then(Value::as_str).is_some_and(|s| s != DEFAULT_IMAGE_SHADER);
    let warning = None;
    let mut range = ng_render::image::default_data_type_range(t);
    if !custom {
        if let Some(r) = o.get("shaderControls").and_then(|c| c.get("normalized")).and_then(|n| n.get("range")) {
            // `parseDataTypeInterval`; an invalid one leaves the default.
            if let Some([a, b]) = r.as_array().map(|a| a.as_slice()).and_then(|a| <&[Value; 2]>::try_from(a).ok()) {
                range = if t == DataType::Uint64 {
                    match (parse_uint64(a), parse_uint64(b)) {
                        (Ok(lo), Ok(hi)) => Interval::Uint64([lo, hi]),
                        _ => range,
                    }
                } else {
                    match (parse_data_type_value(t, a), parse_data_type_value(t, b)) {
                        (Ok(lo), Ok(hi)) => Interval::Number([lo, hi]),
                        _ => range,
                    }
                };
            }
        }
    }
    (opacity, range, warning)
}

/// The `normalized` control's `window` (the CDF panel's bounds): as given,
/// else the default range (`normalizeDataTypeInterval(range)` of the
/// directive's default).
pub fn image_window(o: &Map<String, Value>, t: DataType) -> Interval {
    let default = ng_render::image::default_data_type_range(t);
    let Some([a, b]) = o
        .get("shaderControls")
        .and_then(|c| c.get("normalized"))
        .and_then(|n| n.get("window"))
        .and_then(Value::as_array)
        .and_then(|a| <&[Value; 2]>::try_from(a.as_slice()).ok())
    else {
        return default;
    };
    // `validateDataTypeInterval`: the bounds may not be equal.
    if t == DataType::Uint64 {
        match (parse_uint64(a), parse_uint64(b)) {
            (Ok(lo), Ok(hi)) if lo != hi => Interval::Uint64([lo, hi]),
            _ => default,
        }
    } else {
        match (parse_data_type_value(t, a), parse_data_type_value(t, b)) {
            (Ok(lo), Ok(hi)) if lo != hi => Interval::Number([lo, hi]),
            _ => default,
        }
    }
}

/// A segmentation layer's settings, and its `segments`: (ID, visible).
pub fn segmentation_settings(o: &Map<String, Value>) -> Result<(SegmentationDisplay, Vec<(u64, bool)>)> {
    let d = SegmentationDisplay::default();
    let display = SegmentationDisplay {
        color_seed: o.get("colorSeed").map_or(0, |v| ng_core::SegmentColorHash::from_json(v).seed),
        selected_alpha: alpha(o, "selectedAlpha", d.selected_alpha),
        not_selected_alpha: alpha(o, "notSelectedAlpha", d.not_selected_alpha),
        saturation: alpha(o, "saturation", d.saturation),
        hide_segment_zero: boolean(o, "hideSegmentZero", d.hide_segment_zero),
        hover_highlight: boolean(o, "hoverHighlight", d.hover_highlight),
        ignore_null_visible_set: boolean(o, "ignoreNullVisibleSet", d.ignore_null_visible_set),
        default_color: o.get("segmentDefaultColor").and_then(Value::as_str).and_then(parse_color),
    };
    let mut segments = vec![];
    if let Some(v) = o.get("segments") {
        let a = v.as_array().ok_or_else(|| anyhow!("Expected array, but received: {v}"))?;
        for x in a {
            let s = match x {
                Value::String(s) => s.clone(),
                v => js::to_string(v),
            };
            let (hidden, s) = match s.strip_prefix('!') {
                Some(rest) => (true, rest.to_owned()),
                None => (false, s),
            };
            let id = parse_uint64(&Value::String(s))?;
            // A `Uint64OrderedSet`: the first mention keeps its place.
            match segments.iter_mut().find(|(i, _)| *i == id) {
                Some((_, visible)) => *visible |= !hidden,
                None => segments.push((id, !hidden)),
            }
        }
    }
    Ok((display, segments))
}

/// `crossSectionBackgroundColor`: `#rgb`, `#rrggbb`, or the keyword colours
/// most states use. Others are reported and leave the default.
pub fn parse_color(s: &str) -> Option<[f32; 3]> {
    let hex = |h: &str| u8::from_str_radix(h, 16).ok().map(|v| v as f32 / 255.0);
    let s = s.trim();
    if let Some(h) = s.strip_prefix('#') {
        return match h.len() {
            3 => {
                let c: Vec<f32> = h.chars().map(|c| hex(&format!("{c}{c}"))).collect::<Option<_>>()?;
                Some([c[0], c[1], c[2]])
            }
            6 => Some([hex(&h[0..2])?, hex(&h[2..4])?, hex(&h[4..6])?]),
            _ => None,
        };
    }
    match s.to_ascii_lowercase().as_str() {
        "black" => Some([0.0; 3]),
        "white" => Some([1.0; 3]),
        "gray" | "grey" => Some([128.0 / 255.0; 3]),
        _ => None,
    }
}

/// Viewer state from what a user hands over: a neuroglancer URL or fragment
/// (`…#!{…}`, `#!gs://…/state.json`), JSON (strict or URL-safe), or
/// natively a file of JSON. A remote state is fetched through the kvstores.
pub async fn fetch_state(arg: &str) -> Result<Value> {
    use ng_core::json::{parse_fragment, url_safe_parse, Fragment};
    enum Source {
        Json(Value),
        Text(String),
        Remote(String),
    }
    let arg = arg.trim();
    let source = if let Some(i) = arg.find("#!") {
        match parse_fragment(&arg[i..])? {
            Fragment::State(v) | Fragment::Update(v) => Source::Json(v),
            Fragment::Remote(url) => Source::Remote(url),
            Fragment::Empty => bail!("the URL has no state"),
        }
    } else if arg.starts_with('{') {
        Source::Text(arg.to_owned())
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Source::Text(std::fs::read_to_string(arg).with_context(|| format!("reading {arg}"))?)
        }
        #[cfg(target_arch = "wasm32")]
        {
            bail!("expected viewer state JSON or a URL with #!…: {arg}")
        }
    };
    let v = match source {
        Source::Json(v) => v,
        Source::Text(text) => serde_json::from_str(&text).or_else(|_| url_safe_parse(&text))?,
        Source::Remote(url) => {
            let bytes = ng_kvstore::parse_url(&url)?
                .read("", ng_kvstore::ByteRange::Full)
                .await?
                .with_context(|| format!("{url} not found"))?;
            serde_json::from_slice(&bytes).with_context(|| format!("parsing {url}"))?
        }
    };
    if !v.is_object() {
        bail!("Expected object, but received: {v}");
    }
    Ok(v)
}
