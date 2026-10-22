//! Data source URL completions (`datasource/index.ts`
//! `completeUrl`): with nothing past a scheme yet, the key-value stores
//! neuroglancer offers, and their descriptions.

/// The stores' schemes and descriptions, as neuroglancer lists them.
pub(crate) const KVSTORES: [(&str, &str); 15] = [
    ("boss://", "bossDB: Block & Object Storage System"),
    ("brainmaps://", "Google Brain Maps"),
    ("dvid://", "DVID"),
    ("gs+ngauth+http://", "Google Cloud Storage (ngauth)"),
    ("gs+ngauth+https://", "Google Cloud Storage (ngauth)"),
    ("gs://", "Google Cloud Storage (anonymous)"),
    ("http://", "http (unauthenticated)"),
    ("https://", "https (unauthenticated)"),
    ("local://", "Local in-memory"),
    ("middleauth+https://", "https with middleauth"),
    ("nggraph://", "nggraph data source"),
    ("render://", "Render"),
    ("s3+http://", "S3-compatible http server"),
    ("s3+https://", "S3-compatible https server"),
    ("s3://", "S3 (anonymous)"),
];

/// The completions for `text`: the stores starting with it, until a scheme
/// is typed.
pub(crate) fn completions(text: &str) -> Vec<(&'static str, &'static str)> {
    if text.contains("://") {
        return vec![];
    }
    KVSTORES.iter().copied().filter(|(scheme, _)| scheme.starts_with(text)).collect()
}

/// `suggestLayerName`: the last path component of the last pipeline part
/// with a path (`/` separating, else `:`).
pub(crate) fn suggest_layer_name(url: &str) -> Option<String> {
    for part in url.split('|').rev() {
        // `parsePipelineUrlComponent`: `scheme:` (and `//`), then the suffix.
        let suffix = match part.find(':') {
            Some(i) if part[..i].chars().next().is_some_and(|c| c.is_ascii_alphabetic()) && part[..i].chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) => {
                let rest = &part[i + 1..];
                rest.strip_prefix("//").unwrap_or(rest)
            }
            _ => part,
        };
        if suffix.is_empty() {
            continue;
        }
        // `parseUrlSuffix`: up to `?` or `#`.
        let path = suffix.split(['?', '#']).next().unwrap_or("");
        let path = path.trim_start_matches('/').trim_end_matches('/');
        if path.is_empty() {
            continue;
        }
        let sep = if path.contains('/') { '/' } else { ':' };
        return Some(path.rsplit(sep).next().unwrap_or(path).to_owned());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_schemes() {
        assert_eq!(completions("").len(), 15);
        assert_eq!(completions("gs").len(), 3);
        assert_eq!(completions("s3+").iter().map(|c| c.0).collect::<Vec<_>>(), ["s3+http://", "s3+https://"]);
        assert!(completions("http://x").is_empty());
        assert_eq!(suggest_layer_name("precomputed://http://127.0.0.1:8792/data/image").as_deref(), Some("image"));
        assert_eq!(suggest_layer_name("http://h/a/b/|neuroglancer-precomputed:").as_deref(), Some("b"));
        assert_eq!(suggest_layer_name("gs://bucket").as_deref(), Some("bucket"));
        assert_eq!(suggest_layer_name("zarr:").as_deref(), None);
    }
}
