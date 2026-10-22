//! OME-NGFF multiscale metadata, versions 0.4, 0.5 and 0.6. Port of
//! `src/datasource/zarr/ome.ts` (`parseOmeMetadata` and what it calls),
//! with neuroglancer's matrix arithmetic (`ng_core::matrix`) so transforms
//! agree bit for bit.
//!
//! Not ported yet: `omero` channel metadata (default shader colors).

use anyhow::{anyhow, bail, ensure, Context, Result};
use ng_core::matrix;
use ng_core::si_units::{all_si_prefixes, exp10};
use ng_core::CoordinateSpace;
use serde_json::Value;

/// `OmeMultiscaleScale`.
#[derive(Debug, Clone, PartialEq)]
pub struct OmeScale {
    /// The array's directory URL, ending in `/`.
    pub url: String,
    /// `(rank + 1)²`, column-major: array voxels → model voxels.
    pub transform: Vec<f64>,
}

/// `OmeMultiscaleMetadata`.
#[derive(Debug, Clone)]
pub struct OmeMultiscale {
    pub scales: Vec<OmeScale>,
    pub coordinate_space: CoordinateSpace,
    pub base_scales: Vec<f64>,
    pub base_transform: Vec<f64>,
}

const SUPPORTED_VERSIONS: [&str; 6] = ["0.4", "0.5-dev", "0.5", "0.6.dev1", "0.6.dev3", "0.6"];

/// `OME_UNITS`: `(unit, scale)`.
fn ome_unit(name: &str) -> Option<(&'static str, f64)> {
    Some(match name {
        "angstrom" => ("m", 1e-10),
        "foot" => ("m", 0.3048),
        "inch" => ("m", 0.0254),
        "mile" => ("m", 1609.34),
        // neuroglancer's constant, digit for digit.
        #[allow(clippy::excessive_precision)]
        "parsec" => ("m", 3.0856775814913673e16),
        "yard" => ("m", 0.9144),
        "minute" => ("s", 60.0),
        "hour" => ("s", 3600.0),
        "day" => ("s", 86400.0),
        _ => {
            for (base, unit) in [("meter", "m"), ("second", "s")] {
                if let Some(prefix) = name.strip_suffix(base) {
                    if let Some(p) = all_si_prefixes().find(|p| p.long_prefix == prefix) {
                        return Some((unit, exp10(p.exponent)));
                    }
                }
            }
            return None;
        }
    })
}

struct Axis {
    name: String,
    unit: &'static str,
    scale: f64,
}

/// `parseOmeAxis`.
fn parse_axis(v: &Value) -> Result<Axis> {
    let o = v.as_object().context("axis is not an object")?;
    let name = o.get("name").and_then(Value::as_str).context("axis missing `name`")?;
    let kind = o.get("type").and_then(Value::as_str);
    let (unit, scale) = match o.get("unit") {
        None | Some(Value::Null) => ("", 1.0),
        Some(u) => {
            let s = u.as_str().ok_or_else(|| anyhow!("Unsupported unit: {u}"))?;
            ome_unit(s).ok_or_else(|| anyhow!("Unsupported unit: {s:?}"))?
        }
    };
    let name = if kind == Some("channel") { format!("{name}'") } else { name.to_owned() };
    Ok(Axis { name, unit, scale })
}

fn space_from_axes(axes: &[Axis]) -> CoordinateSpace {
    CoordinateSpace::new(
        true,
        axes.iter().map(|a| a.name.clone()).collect(),
        axes.iter().map(|a| a.unit.to_owned()).collect(),
        axes.iter().map(|a| a.scale).collect(),
    )
}

/// `parseOmeAxes`.
fn parse_axes(v: &Value) -> Result<CoordinateSpace> {
    let axes = v.as_array().context("`axes` is not an array")?.iter().map(parse_axis).collect::<Result<Vec<_>>>()?;
    Ok(space_from_axes(&axes))
}

/// `parseOmeCoordinateSystem`.
fn parse_coordinate_system(v: &Value) -> Result<CoordinateSpace> {
    parse_axes(v.get("axes").context("coordinate system missing `axes`")?)
}

fn floats(v: Option<&Value>, n: usize, what: &str, positive: bool) -> Result<Vec<f64>> {
    let a = v.and_then(Value::as_array).with_context(|| format!("`{what}` is not an array"))?;
    ensure!(a.len() == n, "Expected array of length {n} for `{what}`, but received: {}", a.len());
    a.iter()
        .map(|x| {
            if positive {
                ng_core::js::verify_finite_positive_float(x)
            } else {
                ng_core::js::verify_finite_float(x)
            }
        })
        .collect()
}

fn rows(v: Option<&Value>, nrows: usize, ncols: usize, what: &str) -> Result<Vec<Vec<f64>>> {
    let a = v.and_then(Value::as_array).with_context(|| format!("`{what}` is not an array"))?;
    ensure!(a.len() == nrows, "Expected {what} matrix to have {nrows} rows, but received: {}", a.len());
    a.iter().map(|r| floats(Some(r), ncols, what, false)).collect()
}

/// `parseOmeCoordinateTransform`.
fn parse_transform(rank: usize, v: &Value) -> Result<Vec<f64>> {
    let o = v.as_object().context("coordinate transform is not an object")?;
    let n = rank + 1;
    let kind = o.get("type").and_then(Value::as_str).context("coordinate transform missing `type`")?;
    Ok(match kind {
        "identity" => matrix::identity(n, n),
        "scale" => {
            let s = floats(o.get("scale"), rank, "scale", true)?;
            let mut m = vec![0.0; n * n];
            m[n * n - 1] = 1.0;
            for (i, x) in s.iter().enumerate() {
                m[(n + 1) * i] = *x;
            }
            m
        }
        "translation" => {
            let t = floats(o.get("translation"), rank, "translation", false)?;
            let mut m = matrix::identity(n, n);
            for (i, x) in t.iter().enumerate() {
                m[n * rank + i] = *x;
            }
            m
        }
        "affine" => {
            let a = rows(o.get("affine"), rank, n, "affine")?;
            let mut m = matrix::identity(n, n);
            for i in 0..rank {
                for j in 0..=rank {
                    m[j * n + i] = a[i][j];
                }
            }
            m
        }
        "rotation" => {
            let r = rows(o.get("rotation"), rank, rank, "rotation")?;
            let mut m = matrix::identity(n, n);
            for i in 0..rank {
                for j in 0..rank {
                    m[j * n + i] = r[i][j];
                }
            }
            m
        }
        "mapAxis" => {
            let map = floats(o.get("mapAxis"), rank, "mapAxis", false)?;
            let mut seen = vec![false; rank];
            let mut m = vec![0.0; n * n];
            m[n * n - 1] = 1.0;
            for (i, &x) in map.iter().enumerate() {
                ensure!(
                    x.fract() == 0.0 && x >= 0.0 && (x as usize) < rank,
                    "Invalid mapAxis index: {x}. Must be integer between 0 and {}",
                    rank - 1
                );
                let x = x as usize;
                ensure!(!std::mem::replace(&mut seen[x], true), "Duplicate axis index in mapAxis: {x}");
                // Output[i] = Input[mapAxis[i]].
                m[x * n + i] = 1.0;
            }
            m
        }
        "sequence" => {
            let inner = o.get("transformations");
            if let Some(list) = inner.and_then(Value::as_array) {
                for t in list {
                    ensure!(
                        t.get("type").and_then(Value::as_str) != Some("sequence"),
                        "A sequence transformation MUST NOT be part of another sequence transformation"
                    );
                }
            }
            parse_transforms(rank, inner)?
        }
        other => bail!("Unsupported coordinate transform type: {other:?}"),
    })
}

/// `parseOmeCoordinateTransforms`: composed so the first listed applies first.
fn parse_transforms(rank: usize, v: Option<&Value>) -> Result<Vec<f64>> {
    let n = rank + 1;
    let mut transform = matrix::identity(n, n);
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(transform);
    };
    for t in v.as_array().context("coordinate transforms are not an array")? {
        let next = parse_transform(rank, t)?;
        transform = matrix::multiply(&next, &transform, n);
    }
    Ok(transform)
}

/// `parseOmeInputOutput`: an object `{name, path}` or, in RFC drafts, a string
/// standing for both.
fn input_output(v: &Value) -> Result<(Option<String>, Option<String>)> {
    match v {
        Value::Object(o) => Ok((
            o.get("name").and_then(Value::as_str).map(str::to_owned),
            o.get("path").and_then(Value::as_str).map(str::to_owned),
        )),
        Value::String(s) => Ok((Some(s.clone()), Some(s.clone()))),
        _ => bail!("Expected input/output to be string or object"),
    }
}

/// `validateCoordinateTransformations` (0.6).
fn validate_transformations(transformations: Option<&Value>, expected_input: &str, expected_output: &str, path: &str) -> Result<()> {
    let Some(list) = transformations.and_then(Value::as_array) else { return Ok(()) };
    if list.len() != 1 {
        return Ok(());
    }
    let t = &list[0];
    ensure!(t.is_object(), "coordinate transform is not an object");
    let input = t.get("input").map(input_output).transpose()?;
    if let Some((_, Some(p))) = &input {
        ensure!(
            p.is_empty() || p == expected_input,
            "Invalid coordinate transformation for dataset at path \"{path}\": input is \"{p}\" but expected \"{expected_input}\""
        );
    }
    if t.get("type").and_then(Value::as_str) == Some("sequence") {
        if let Some(inner) = t.get("transformations").and_then(Value::as_array) {
            for (i, it) in inner.iter().enumerate() {
                let inner_input = it.get("input").map(input_output).transpose()?;
                let inner_output = it.get("output").map(input_output).transpose()?;
                if i == 0 {
                    if let Some((_, Some(p))) = &inner_input {
                        ensure!(p == expected_input, "Invalid sequence transformation for dataset at path \"{path}\": first inner transform has input \"{p}\" but expected \"{expected_input}\"");
                    }
                }
                if i + 1 == inner.len() {
                    if let Some((Some(name), _)) = &inner_output {
                        ensure!(name == expected_output, "Invalid sequence transformation for dataset at path \"{path}\": last inner transform has output \"{name}\" but expected \"{expected_output}\"");
                    }
                }
                if i > 0 {
                    let prev_output = inner[i - 1].get("output").map(input_output).transpose()?;
                    if let (Some((Some(a), _)), Some((Some(b), _))) = (&prev_output, &inner_input) {
                        ensure!(a == b, "Invalid sequence transformation for dataset at path \"{path}\": transform {} has output \"{a}\" but transform {i} has input \"{b}\". Transforms in a sequence must have matching input/output for consecutive transforms.", i - 1);
                    }
                }
            }
        }
    }
    Ok(())
}

/// `extractScalesFromAffineMatrix`: the length of each column.
pub fn extract_scales(t: &[f64], rank: usize) -> Vec<f64> {
    (0..rank)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..rank {
                let v = t[i * (rank + 1) + j];
                s += v * v;
            }
            s.sqrt()
        })
        .collect()
}

/// `kvstoreEnsureDirectoryPipelineUrl(joinBaseUrlAndPath(base, path))`: plain
/// concatenation (the base is a directory URL, ending in `/`), then a
/// trailing `/`.
fn scale_url(base: &str, path: &str) -> String {
    let mut url = format!("{base}{}", ng_kvstore::encode_path_for_url(path));
    if !url.ends_with('/') {
        url.push('/');
    }
    url
}

/// `parseOmeMultiscale`.
fn parse_multiscale(url: &str, multiscale: &Value, version: &str) -> Result<OmeMultiscale> {
    let o = multiscale.as_object().context("multiscale is not an object")?;
    let datasets = o.get("datasets").and_then(Value::as_array).context("multiscale missing `datasets`")?;
    let systems = o.get("coordinateSystems").and_then(Value::as_array).filter(|a| !a.is_empty());
    let transforms_json = o.get("coordinateTransformations");
    let mut transforms_to_apply = transforms_json.cloned();
    let mut intrinsic: Option<String> = None;

    let mut coordinate_space = if let Some(systems) = systems {
        // 0.6+: the intrinsic system is the one every dataset transform outputs to.
        let outputs = datasets
            .iter()
            .map(|d| {
                let first = d
                    .get("coordinateTransformations")
                    .and_then(Value::as_array)
                    .and_then(|a| a.first())
                    .and_then(|t| t.get("output"))
                    .unwrap_or(&Value::Null);
                input_output(first).map(|(name, _)| name)
            })
            .collect::<Result<Vec<_>>>()?;
        intrinsic = outputs.first().cloned().flatten();
        if let Some(i) = outputs.iter().position(|x| *x != intrinsic) {
            bail!(
                "All output names of multiscale.datasets must be the same, candidate name from first scale is {:?}, but found {:?} at scale {i}",
                intrinsic,
                outputs[i]
            );
        }
        let intrinsic_name = intrinsic.clone().context("There must be an intrinsic coordinateSystem")?;
        let parsed = systems.iter().map(parse_coordinate_system).collect::<Result<Vec<_>>>()?;
        // Transforms out of the intrinsic system; neuroglancer applies the first.
        let mut applicable = vec![];
        if let Some(list) = transforms_json.and_then(Value::as_array) {
            for t in list {
                let input = t.get("input").map(input_output).transpose()?;
                let output = t.get("output").map(input_output).transpose()?;
                if input.as_ref().and_then(|x| x.0.as_deref()) != Some(intrinsic_name.as_str()) {
                    continue;
                }
                let Some((Some(out_name), out_path)) = output else { continue };
                if let Some(p) = &out_path {
                    if p.contains("labels") && *p != out_name {
                        continue;
                    }
                }
                applicable.push((t.clone(), out_name));
            }
        }
        transforms_to_apply = applicable.first().map(|(t, _)| Value::Array(vec![t.clone()]));
        let name_to_match = applicable.first().map(|(_, n)| n.clone()).unwrap_or_else(|| intrinsic_name.clone());
        let i = systems
            .iter()
            .position(|s| s.get("name").and_then(Value::as_str) == Some(name_to_match.as_str()))
            .with_context(|| {
                let reason = if name_to_match == intrinsic_name { "intrinsic" } else { "output" };
                format!("Could not find any coordinate system for the {reason} system {name_to_match}")
            })?;
        parsed[i].clone()
    } else {
        // 0.4 / 0.5.
        parse_axes(o.get("axes").context("multiscale missing `axes`")?)?
    };

    let rank = coordinate_space.rank();
    let n = rank + 1;
    let transform = parse_transforms(rank, transforms_to_apply.as_ref())?;
    let mut scales = datasets
        .iter()
        .map(|d| {
            let path = d.get("path").and_then(Value::as_str).context("dataset missing `path`")?;
            let transformations = d.get("coordinateTransformations");
            if let Some(name) = &intrinsic {
                validate_transformations(transformations, path, name, path)?;
            }
            let t = parse_transforms(rank, transformations)?;
            Ok(OmeScale {
                url: scale_url(url, path),
                transform: matrix::multiply(&transform, &t, n),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!scales.is_empty(), "At least one scale must be specified");

    let base_transform = scales[0].transform.clone();
    let base_scales = extract_scales(&base_transform, rank);
    for (s, b) in coordinate_space.scales.iter_mut().zip(&base_scales) {
        *s *= b;
    }
    // OME puts a voxel's origin at its center; neuroglancer at its lower
    // corner. Shift by half a voxel at each scale.
    for scale in &mut scales {
        let t = &mut scale.transform;
        for i in 0..rank {
            let mut offset = 0.0;
            for j in 0..rank {
                offset += t[j * n + i] * 0.5;
            }
            t[rank * n + i] -= offset;
        }
    }

    let old_behavior = matches!(version, "0.4" | "0.5-dev" | "0.5");
    if old_behavior {
        // Identity base transform; per-scale transforms divided by the base
        // scales row by row.
        for scale in &mut scales {
            let t = &mut scale.transform;
            for i in 0..rank {
                for j in 0..=rank {
                    t[j * n + i] /= base_scales[i];
                }
            }
        }
        Ok(OmeMultiscale {
            scales,
            coordinate_space,
            base_scales,
            base_transform: matrix::identity(n, n),
        })
    } else {
        // 0.6: per-scale transforms relative to the base; the base, with its
        // scale divided out column by column, is the model transform.
        let (inverse_base, _) = matrix::inverse(&base_transform, n);
        let mut base_without_scale = base_transform.clone();
        for i in 0..rank {
            for j in 0..rank {
                base_without_scale[i * n + j] /= base_scales[i];
            }
            base_without_scale[rank * n + i] /= base_scales[i];
        }
        for scale in &mut scales {
            scale.transform = matrix::multiply(&inverse_base, &scale.transform, n);
        }
        Ok(OmeMultiscale {
            scales,
            coordinate_space,
            base_scales,
            base_transform: base_without_scale,
        })
    }
}

/// `parseOmeMetadata`: `None` if `attrs` holds no OME multiscale.
pub fn parse(url: &str, attrs: &Value, zarr_version: u8) -> Result<Option<OmeMultiscale>> {
    let ome = attrs.get("ome");
    let metadata = ome.unwrap_or(attrs);
    let Some(multiscales) = metadata.get("multiscales").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut errors = vec![];
    for multiscale in multiscales {
        if !multiscale.is_object() {
            return Ok(None);
        }
        let version = ome
            .and_then(|o| o.get("version"))
            .or_else(|| multiscale.get("version"))
            .and_then(Value::as_str);
        let Some(version) = version else { return Ok(None) };
        if !SUPPORTED_VERSIONS.contains(&version) {
            errors.push(format!("OME multiscale metadata version {version:?} is not supported"));
            continue;
        }
        if version != "0.4" && version != "0.5-dev" && zarr_version != 3 {
            errors.push(format!(
                "OME multiscale metadata version {version:?} is not supported for zarr v{zarr_version}"
            ));
            continue;
        }
        return parse_multiscale(url, multiscale, version).map(Some);
    }
    match errors.into_iter().next() {
        Some(e) => Err(anyhow!(e)),
        None => Ok(None),
    }
}
