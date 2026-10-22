//! Tool key bindings (`ui/tool.ts`: `GlobalToolBinder`, `LocalToolBinder`):
//! letters `A`–`Z` bound to tools, each tool's JSON kept by the binder it
//! belongs to — the viewer's (`toolBindings`, e.g. dimension tools) or a
//! layer's (its `toolBindings`, e.g. its controls). A key binds one tool
//! across all binders, and a tool has at most one key in its binder.

use serde_json::{Map, Value};

use crate::Viewer;

/// The binder a tool belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolOwner {
    /// The viewer's (`viewer.toolBinder`).
    Global,
    /// Layer `i`'s (`layer.toolBinder`).
    Layer(usize),
}

/// A bound tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolBinding {
    pub key: char,
    pub owner: ToolOwner,
    pub json: Value,
}

/// `TOOL_KEY_PATTERN`.
pub fn is_tool_key(key: &str) -> bool {
    key.len() == 1 && key.as_bytes()[0].is_ascii_uppercase()
}

/// A binder's `toolBindings` object: keys as bound, in binding order.
fn bindings_of(json: Option<&Value>) -> Vec<(char, Value)> {
    let Some(Value::Object(o)) = json else { return vec![] };
    o.iter().filter(|(k, _)| is_tool_key(k)).map(|(k, v)| (k.chars().next().unwrap(), v.clone())).collect()
}

fn to_json(bindings: &[(char, Value)]) -> Option<Value> {
    if bindings.is_empty() {
        return None;
    }
    let mut o = Map::new();
    for (k, v) in bindings {
        o.insert(k.to_string(), v.clone());
    }
    Some(Value::Object(o))
}

impl Viewer {
    fn owner_bindings(&self, owner: ToolOwner) -> Vec<(char, Value)> {
        match owner {
            ToolOwner::Global => bindings_of(self.ui_state("toolBindings")),
            ToolOwner::Layer(i) => bindings_of(self.layers().get(i).and_then(|l| l.spec.json.get("toolBindings"))),
        }
    }

    fn set_owner_bindings(&mut self, owner: ToolOwner, bindings: &[(char, Value)]) {
        match owner {
            ToolOwner::Global => self.set_ui_state("toolBindings", to_json(bindings)),
            ToolOwner::Layer(i) => self.set_layer_option(i, "toolBindings", to_json(bindings)),
        }
    }

    fn owners(&self) -> Vec<ToolOwner> {
        std::iter::once(ToolOwner::Global).chain((0..self.layers().len()).map(ToolOwner::Layer)).collect()
    }

    /// Every bound tool.
    pub fn tool_bindings(&self) -> Vec<ToolBinding> {
        self.owners()
            .into_iter()
            .flat_map(|owner| self.owner_bindings(owner).into_iter().map(move |(key, json)| ToolBinding { key, owner, json }))
            .collect()
    }

    /// The tool bound to `key`.
    pub fn tool(&self, key: char) -> Option<ToolBinding> {
        self.tool_bindings().into_iter().find(|b| b.key == key)
    }

    /// The key of tool `json` in `owner`'s binder (`jsonToKey`).
    pub fn tool_key(&self, owner: ToolOwner, json: &Value) -> Option<char> {
        self.owner_bindings(owner).into_iter().find(|(_, v)| v == json).map(|(k, _)| k)
    }

    /// `GlobalToolBinder.set`: binds `key` to tool `json` of `owner`,
    /// unbinding what `key` was bound to and the key the tool had.
    pub fn bind_tool(&mut self, key: char, owner: ToolOwner, json: Value) {
        for o in self.owners() {
            let mut bindings = self.owner_bindings(o);
            let n = bindings.len();
            bindings.retain(|(k, v)| *k != key && !(o == owner && *v == json));
            if bindings.len() != n {
                self.set_owner_bindings(o, &bindings);
            }
        }
        let mut bindings = self.owner_bindings(owner);
        bindings.push((key, json));
        self.set_owner_bindings(owner, &bindings);
    }

    /// `LocalToolBinder.removeJsonString`: unbinds tool `json` of `owner`.
    pub fn unbind_tool(&mut self, owner: ToolOwner, json: &Value) {
        let mut bindings = self.owner_bindings(owner);
        let n = bindings.len();
        bindings.retain(|(_, v)| v != json);
        if bindings.len() != n {
            self.set_owner_bindings(owner, &bindings);
        }
    }
}

/// The image layer's controls (`LAYER_CONTROLS`), as tool types.
const IMAGE_CONTROLS: [&str; 4] = ["crossSectionRenderScale", "blend", "opacity", "volumeRendering"];
/// Volume rendering's, valid while it is on.
const VOLUME_CONTROLS: [&str; 2] = ["volumeRenderingGain", "volumeRenderingDepthSamples"];
/// The segmentation layer's 2D controls and checkboxes (its 3D and
/// skeleton ones need meshes and skeletons).
const SEGMENTATION_CONTROLS: [&str; 10] = [
    "colorSeed",
    "segmentDefaultColor",
    "saturation",
    "selectedAlpha",
    "notSelectedAlpha",
    "crossSectionRenderScale",
    "hideSegmentZero",
    "baseSegmentColoring",
    "ignoreNullVisibleSet",
    "hoverHighlight",
];

/// The `#uicontrol` names of a shader (`ShaderControlState`'s keys).
pub fn shader_control_names(shader: &str) -> Vec<String> {
    shader
        .lines()
        .filter_map(|l| {
            let mut words = l.trim_start().strip_prefix("#uicontrol")?.split_whitespace();
            let _type = words.next()?;
            Some(words.next()?.trim_end_matches(|c: char| !(c.is_alphanumeric() || c == '_')).to_owned())
        })
        .collect()
}

impl Viewer {
    /// The tools each binder lists (`ToolLister`s), as palette JSON with
    /// the binder's common properties (`layer`, `layerType`), binders in
    /// `getSortOrder` order: the viewer's (its dimensions), then each
    /// layer's.
    pub fn tool_lists(&self) -> Vec<Vec<Value>> {
        let mut out = vec![self.dimensions().names.iter().map(|n| serde_json::json!({ "type": "dimension", "dimension": n })).collect::<Vec<_>>()];
        for l in self.layers() {
            let name = l.name().to_owned();
            let layer_type = l.spec.json.get("type").and_then(Value::as_str).unwrap_or("").to_owned();
            let mut types: Vec<Value> = vec![];
            match layer_type.as_str() {
                "image" => {
                    types.extend(IMAGE_CONTROLS.iter().map(|&t| Value::from(t)));
                    let volume = l.spec.json.get("volumeRendering").is_some_and(|v| v.as_bool() == Some(true) || v.as_str().is_some_and(|s| s != "off"));
                    if volume {
                        types.extend(VOLUME_CONTROLS.iter().map(|&t| Value::from(t)));
                    }
                    let shader = l.spec.json.get("shader").and_then(Value::as_str).unwrap_or(crate::state::DEFAULT_IMAGE_SHADER);
                    types.extend(shader_control_names(shader).into_iter().map(|c| serde_json::json!({ "type": "shaderControl", "control": c })));
                }
                "segmentation" => types.extend(SEGMENTATION_CONTROLS.iter().map(|&t| Value::from(t))),
                _ => {}
            }
            out.push(
                types
                    .into_iter()
                    .map(|t| {
                        let mut o = Map::new();
                        o.insert("layer".into(), name.clone().into());
                        match t {
                            Value::String(s) => {
                                o.insert("type".into(), s.into());
                            }
                            Value::Object(m) => o.extend(m),
                            _ => {}
                        }
                        o.insert("layerType".into(), layer_type.clone().into());
                        Value::Object(o)
                    })
                    .collect(),
            );
        }
        out
    }

    /// `getToolFromJson`: a palette's tool JSON as its binder and local
    /// JSON (`{layer, type}` → layer, `"type"`); `None` for a layer that
    /// does not exist.
    pub fn tool_from_palette_json(&self, json: &Value) -> Option<(ToolOwner, Value)> {
        let o = json.as_object()?;
        let Some(layer) = o.get("layer").and_then(Value::as_str) else { return Some((ToolOwner::Global, json.clone())) };
        let i = self.layers().iter().position(|l| l.name() == layer)?;
        let mut local = o.clone();
        local.remove("layer");
        local.remove("layerType");
        // Layer controls and the select tool are named by their type alone.
        let local = if local.len() == 1 { local.get("type").cloned()? } else { Value::Object(local) };
        Some((ToolOwner::Layer(i), local))
    }

    /// `convertLocalJSONToPaletteJSON`.
    pub fn palette_json(&self, owner: ToolOwner, json: &Value) -> Value {
        match owner {
            ToolOwner::Global => json.clone(),
            ToolOwner::Layer(i) => {
                let mut o = Map::new();
                o.insert("layer".into(), self.layers().get(i).map_or("", |l| l.name()).into());
                match json {
                    Value::String(s) => {
                        o.insert("type".into(), s.clone().into());
                    }
                    Value::Object(m) => o.extend(m.clone()),
                    _ => {}
                }
                Value::Object(o)
            }
        }
    }
}

impl Viewer {
    /// `DimensionTool`s follow their dimension's rename (`toolJsonMaybeChanged`):
    /// in the viewer's bindings and the tool palettes.
    pub(crate) fn rename_dimension_tools(&mut self, old: &[String], new: &[String]) {
        let rename = |v: &mut Value| {
            if v.get("type").and_then(Value::as_str) != Some("dimension") || v.get("layer").is_some() {
                return;
            }
            let Some(name) = v.get("dimension").and_then(Value::as_str) else { return };
            if let Some(i) = old.iter().position(|o| o == name) {
                if let Some(o) = v.as_object_mut() {
                    o.insert("dimension".into(), new[i].clone().into());
                }
            }
        };
        let mut bindings = self.owner_bindings(ToolOwner::Global);
        for (_, v) in &mut bindings {
            rename(v);
        }
        self.set_owner_bindings(ToolOwner::Global, &bindings);
        if let Some(Value::Object(mut palettes)) = self.ui_state("toolPalettes").cloned() {
            for p in palettes.values_mut() {
                if let Some(Value::Array(tools)) = p.get_mut("tools") {
                    tools.iter_mut().for_each(rename);
                }
            }
            self.set_ui_state("toolPalettes", Some(Value::Object(palettes)));
        }
    }
}
