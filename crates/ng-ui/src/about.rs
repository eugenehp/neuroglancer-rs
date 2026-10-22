//! The port and who made it, as its package metadata says (the workspace
//! `Cargo.toml`): shown in the Settings panel and the footer.

/// The port's name.
pub const NAME: &str = "neuroglancer-rs";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The authors, `:`-separated as Cargo gives them.
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const LICENSE: &str = env!("CARGO_PKG_LICENSE");
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");
/// The copyright year.
pub const YEAR: &str = "2026";

/// `© 2026 Eugene Hauptmann`.
pub fn copyright() -> String {
    format!("© {YEAR} {}", AUTHORS.split(':').collect::<Vec<_>>().join(", "))
}

/// The repository without its scheme: `github.com/…`.
pub fn repository_label() -> &'static str {
    REPOSITORY.trim_start_matches("https://").trim_start_matches("http://")
}

/// The footer's line: name, copyright, license, repository.
pub fn footer() -> String {
    format!("{NAME} · {} · {LICENSE} · {}", copyright(), repository_label())
}

#[cfg(test)]
mod tests {
    #[test]
    fn credits_come_from_the_package_metadata() {
        assert_eq!(super::copyright(), "© 2026 Eugene Hauptmann");
        assert_eq!(super::LICENSE, "MIT OR Apache-2.0");
        assert_eq!(super::repository_label(), "github.com/eugenehp/neuroglancer-rs");
        assert_eq!(super::footer(), "neuroglancer-rs · © 2026 Eugene Hauptmann · MIT OR Apache-2.0 · github.com/eugenehp/neuroglancer-rs");
    }
}
