//! The statistics panel's source names (`ui/statistics.ts`
//! `getFormattedNames`): each source's identifying properties, only those
//! that tell it from the others.

/// `getDistinguishingProperties` then `getNameFromProps`: per source, the
/// JSON object of the properties needed (`.type` always) to tell it apart.
pub(crate) fn formatted_names(properties: &[Vec<(String, String)>]) -> Vec<String> {
    let get = |i: usize, p: &str| properties[i].iter().find(|(k, _)| k == p).map(|(_, v)| v.as_str());
    let mut selected: Vec<String> = vec![".type".into()];
    let mut all: Vec<String> = vec![];
    for (i, props) in properties.iter().enumerate() {
        for (k, _) in props {
            if !all.contains(k) {
                all.push(k.clone());
            }
        }
        let distinguished = |selected: &[String], j: usize| selected.iter().any(|p| get(i, p) != get(j, p));
        let mut matches: Vec<usize> = (0..i).filter(|&j| !distinguished(&selected, j)).collect();
        while !matches.is_empty() {
            let mut best = matches.clone();
            let mut best_prop: Option<String> = None;
            for p in &all {
                if selected.contains(p) {
                    continue;
                }
                let reduced: Vec<usize> = matches.iter().copied().filter(|&j| get(j, p) == get(i, p)).collect();
                if reduced.len() < best.len() {
                    best = reduced.clone();
                    best_prop = Some(p.clone());
                }
                if reduced.is_empty() {
                    break;
                }
            }
            let Some(p) = best_prop else { break };
            matches = best;
            selected.push(p);
        }
    }
    (0..properties.len())
        .map(|i| {
            let mut o = serde_json::Map::new();
            for p in &selected {
                if let Some(v) = get(i, p) {
                    if p.is_empty() {
                        return v.to_owned();
                    }
                    o.insert(p.clone(), v.into());
                }
            }
            serde_json::Value::Object(o).to_string()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_as_neuroglancer() {
        let p = |url: &str| vec![(".type".to_owned(), "precomputed/VolumeChunkSource".to_owned()), (".parameters.url".to_owned(), url.to_owned())];
        let names = formatted_names(&[p("http://h/image/4_4_40/"), p("http://h/seg/8_8_80/")]);
        assert_eq!(names[0], r#"{".type":"precomputed/VolumeChunkSource",".parameters.url":"http://h/image/4_4_40/"}"#);
        assert_eq!(formatted_names(&[p("a")]), vec![r#"{".type":"precomputed/VolumeChunkSource"}"#]);
    }
}
