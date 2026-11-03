//! Data source URLs, as neuroglancer accepts them:
//!
//! - the pipeline form: `<kvstore url>|<adapter>:…|<format>:`, e.g.
//!   `gs://bucket/vol|neuroglancer-precomputed:`, `https://h/a.zarr|zarr3:`,
//!   `https://h/x.nii.gz|gzip:|nifti:`;
//! - the legacy form: `precomputed://<kvstore url>`, `zarr://…`, `zarr2://…`,
//!   `zarr3://…`, `nifti://…`;
//! - and (here) a bare kvstore URL or local path, whose format is detected:
//!   NIfTI by extension, precomputed by an `info` file, Zarr by its metadata,
//!   N5 by `attributes.json`, Deep Zoom by a `.dzi` file's XML.

use anyhow::{bail, Result};
use ng_kvstore::{parse_url, ByteRange, KvStoreLocation};

use crate::model::DataSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Precomputed,
    Zarr,
    Nifti,
    N5,
    DeepZoom,
}

/// A parsed data source URL: where, and what format (if stated).
#[derive(Debug, Clone)]
pub struct DataSourceUrl {
    /// The kvstore URL.
    pub kvstore: String,
    pub format: Option<Format>,
    /// `zarr2://` / `zarr3://` name the Zarr version: only its metadata is read.
    pub zarr_version: Option<u8>,
}

fn zarr_version(scheme: &str) -> Option<u8> {
    match scheme {
        "zarr2" => Some(2),
        "zarr3" => Some(3),
        _ => None,
    }
}

fn format_scheme(s: &str) -> Option<Format> {
    match s {
        "precomputed" | "neuroglancer-precomputed" => Some(Format::Precomputed),
        "zarr" | "zarr2" | "zarr3" => Some(Format::Zarr),
        "nifti" => Some(Format::Nifti),
        "n5" => Some(Format::N5),
        "deepzoom" => Some(Format::DeepZoom),
        _ => None,
    }
}

pub fn parse(url: &str) -> Result<DataSourceUrl> {
    let url = url.trim();
    // Pipeline form: the last `|format:` names the format; the base URL and
    // the kvstore adapters before it (`zip:`, `gzip:`) the kvstore.
    if url.contains('|') {
        let parts: Vec<&str> = url.split('|').collect();
        let scheme = |part: &str| part.split(':').next().unwrap_or(part).to_owned();
        let format_at = (1..parts.len()).rev().find(|&i| format_scheme(&scheme(parts[i])).is_some());
        let (kvstore, format, version) = match format_at {
            Some(i) => (parts[..i].join("|"), format_scheme(&scheme(parts[i])), zarr_version(&scheme(parts[i]))),
            None => (url.to_owned(), None, None),
        };
        return Ok(DataSourceUrl { kvstore, format, zarr_version: version });
    }
    // Legacy form.
    if let Some((scheme, rest)) = url.split_once("://") {
        if let Some(format) = format_scheme(scheme) {
            return Ok(DataSourceUrl {
                kvstore: rest.to_owned(),
                format: Some(format),
                zarr_version: zarr_version(scheme),
            });
        }
    }
    Ok(DataSourceUrl {
        kvstore: url.to_owned(),
        format: None,
        zarr_version: None,
    })
}

/// A kvstore location for `url`; a bare local path is a `file://` URL.
pub fn location(url: &str) -> Result<KvStoreLocation> {
    if url.contains('|') {
        ng_kvstore::parse_pipeline(url)
    } else if url.contains("://") {
        parse_url(url)
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use anyhow::Context;
            let abs = std::path::absolute(url).with_context(|| format!("resolving {url}"))?;
            parse_url(&format!("file://{}", abs.display()))
        }
        #[cfg(target_arch = "wasm32")]
        bail!("not a URL: {url:?}")
    }
}

/// Opens the data source `url` names.
#[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
pub async fn get(url: &str) -> Result<DataSource> {
    if url.trim() == LOCAL_ANNOTATIONS {
        return Ok(local_annotations());
    }
    let parsed = parse(url)?;
    let mut loc = location(&parsed.kvstore)?;
    let format = match parsed.format {
        Some(f) => f,
        None => {
            // A zip archive (`PK\x03\x04`): what is inside it.
            let lower = parsed.kvstore.to_ascii_lowercase();
            if !lower.ends_with('/') && !lower.ends_with(".nii") && !lower.ends_with(".nii.gz") && !lower.ends_with(".dzi") && is_zip(&loc).await? {
                loc = KvStoreLocation { store: std::sync::Arc::new(ng_kvstore::ZipStore::new(loc)), path: String::new() };
            }
            detect(&parsed.kvstore, &loc).await?
        }
    };
    match format {
        Format::Nifti => {
            let mut shown = loc.url();
            if shown.ends_with(".gz") {
                shown.push_str("|gzip:");
            }
            crate::nifti::get(&loc, &shown).await
        }
        Format::Precomputed => crate::precomputed::get(&loc).await,
        Format::Zarr => crate::zarr::get_version(&loc, parsed.zarr_version).await,
        Format::N5 => crate::n5::get(&loc).await,
        Format::DeepZoom => crate::deepzoom::get(&loc).await,
    }
}

/// `localAnnotationsUrl`.
pub const LOCAL_ANNOTATIONS: &str = "local://annotations";

/// `LocalDataSourceProvider` for annotations: no model space of its own
/// (the viewer gives it the global one when the state names no transform).
pub fn local_annotations() -> DataSource {
    let empty = std::sync::Arc::new(ng_core::CoordinateSpace::new(true, vec![], vec![], vec![]));
    DataSource {
        canonical_url: LOCAL_ANNOTATIONS.into(),
        model_transform: crate::model::CoordinateSpaceTransform::identity(empty),
        subsources: vec![crate::model::SubsourceEntry { id: "default".into(), default: true, subsource: crate::model::Subsource::LocalAnnotations }],
    }
}

/// Whether `loc` is a file starting as a zip archive does.
async fn is_zip(loc: &KvStoreLocation) -> Result<bool> {
    Ok(loc.read("", ByteRange::Range { offset: 0, length: 4 }).await?.is_some_and(|b| b == [0x50, 0x4b, 0x03, 0x04]))
}

async fn detect(url: &str, loc: &KvStoreLocation) -> Result<Format> {
    let lower = url.to_ascii_lowercase();
    if lower.ends_with(".nii") || lower.ends_with(".nii.gz") {
        return Ok(Format::Nifti);
    }
    // A file starting as a Deep Zoom image does.
    if lower.ends_with(".dzi") {
        if let Some(prefix) = loc.read("", ByteRange::Range { offset: 0, length: 500 }).await? {
            if crate::deepzoom::detect(&prefix) {
                return Ok(Format::DeepZoom);
            }
        }
    }
    let dir = loc.as_directory();
    if dir.read("info", ByteRange::Full).await?.is_some() {
        return Ok(Format::Precomputed);
    }
    for key in ["zarr.json", ".zarray", ".zattrs"] {
        if dir.read(key, ByteRange::Full).await?.is_some() {
            return Ok(Format::Zarr);
        }
    }
    if dir.read("attributes.json", ByteRange::Full).await?.is_some() {
        return Ok(Format::N5);
    }
    bail!("could not tell what kind of data is at {url}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zarr_scheme_names_the_version() {
        assert_eq!(parse("zarr3://https://h/a").unwrap().zarr_version, Some(3));
        assert_eq!(parse("zarr2://https://h/a").unwrap().zarr_version, Some(2));
        assert_eq!(parse("zarr://https://h/a").unwrap().zarr_version, None);
        assert_eq!(parse("https://h/a|zarr3:").unwrap().zarr_version, Some(3));
    }

    #[test]
    fn url_forms() {
        let u = parse("precomputed://gs://b/v").unwrap();
        assert_eq!((u.kvstore.as_str(), u.format), ("gs://b/v", Some(Format::Precomputed)));
        let u = parse("https://h/a.zarr|zarr3:").unwrap();
        assert_eq!((u.kvstore.as_str(), u.format), ("https://h/a.zarr", Some(Format::Zarr)));
        let u = parse("https://h/x.nii.gz|gzip:|nifti:").unwrap();
        assert_eq!((u.kvstore.as_str(), u.format), ("https://h/x.nii.gz|gzip:", Some(Format::Nifti)));
        let u = parse("https://h/a.zip|zip:x/|zarr3:").unwrap();
        assert_eq!((u.kvstore.as_str(), u.format), ("https://h/a.zip|zip:x/", Some(Format::Zarr)));
        let u = parse("s3://b/d").unwrap();
        assert_eq!((u.kvstore.as_str(), u.format), ("s3://b/d", None));
    }
}
