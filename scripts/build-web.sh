#!/usr/bin/env bash
# Builds ng-web into crates/ng-web/www/pkg. Serve crates/ng-web/www/ afterwards,
# e.g. `python3 -m http.server -d crates/ng-web/www 8080`.
#
# Requires: rustup target add wasm32-unknown-unknown, and a wasm-bindgen CLI
# whose version equals the `wasm-bindgen` crate in Cargo.lock (kept on threers'
# version). PROFILE=release-wasm for a small build.
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${PROFILE:-dev}"
cargo build -p ng-web --target wasm32-unknown-unknown --profile "$profile"

dir="$profile"
[ "$profile" = dev ] && dir=debug
want=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/, ""); print}' Cargo.lock)
have=$(wasm-bindgen --version | awk '{print $2}')
if [ "$want" != "$have" ]; then
  echo "wasm-bindgen CLI is $have but Cargo.lock has $want:" >&2
  echo "  cargo install wasm-bindgen-cli --version $want" >&2
  exit 1
fi
wasm-bindgen --target web --no-typescript \
  --out-dir crates/ng-web/www/pkg \
  "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/$dir/ng_web.wasm"
echo "built crates/ng-web/www/pkg ($profile)"
