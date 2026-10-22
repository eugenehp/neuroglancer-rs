//! The engine stays reusable without the UI: no engine crate depends on
//! `ng-ui` or on a shell, and `ng-ui` depends on no shell.

const ENGINE: [&str; 6] = ["ng-core", "ng-kvstore", "ng-chunk", "ng-datasource", "ng-render", "ng-viewer"];

fn dependencies(krate: &str) -> String {
    let path = format!("{}/../{krate}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    // Everything from the first dependency table on.
    manifest.find("dependencies]").map_or(String::new(), |i| manifest[i..].to_owned())
}

fn names(deps: &str) -> Vec<String> {
    deps.lines()
        .filter_map(|l| l.split(['=', '.']).next())
        .map(|n| n.trim().to_owned())
        .filter(|n| n.starts_with("ng-"))
        .collect()
}

#[test]
fn engine_crates_do_not_depend_on_the_ui_or_shells() {
    for krate in ENGINE {
        for dep in names(&dependencies(krate)) {
            assert!(ENGINE.contains(&dep.as_str()), "{krate} depends on {dep}, outside the engine");
        }
    }
}

#[test]
fn the_ui_depends_on_no_shell() {
    for dep in names(&dependencies("ng-ui")) {
        assert!(ENGINE.contains(&dep.as_str()), "ng-ui depends on {dep}");
    }
}
