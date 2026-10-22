#!/usr/bin/env bash
# Builds neuroglancer (from a copy of ../neuroglancer, so the checkout gets no
# node_modules or build output) and serves it, neuroglancer-rs's web build and
# the test data from one origin, so `compare.mjs` can render the same state in
# both. Leaves the server running on $PORT (default 8792).
#
# Requires node/npm (npm ≥ 11 needs --allow-git=all for neuroglancer's git
# dependency) and a Playwright Chromium (see compare.mjs).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
ng="${NEUROGLANCER:-$root/../neuroglancer}"
work="${WORK:-/tmp/ng-ref}"
serve="${SERVE:-/tmp/ng-serve}"
port="${PORT:-8792}"

mkdir -p "$work"
rsync -a --delete --exclude .git --exclude node_modules --exclude dist "$ng/" "$work/"
(cd "$work" && npm ci --no-audit --no-fund --allow-git=all >/dev/null && npm run build >/dev/null)

"$root/scripts/build-web.sh"

mkdir -p "$serve"
ln -sfn "$work/dist/client" "$serve/ng"
ln -sfn "$root/crates/ng-web/www" "$serve/rs"
ln -sfn "$root/crates/ng-viewer/tests/fixtures" "$serve/data"
ln -sfn "$ng/testdata" "$serve/testdata"
if ! curl -s -o /dev/null "http://127.0.0.1:$port/ng/index.html"; then
  (cd "$serve" && nohup python3 "$root/scripts/reference/serve.py" "$port" >/tmp/ng-serve.log 2>&1 &)
  sleep 1
fi
echo "serving $serve on http://127.0.0.1:$port"
