"""Where each text run lands, neuroglancer against neuroglancer-rs, at full
precision: every run of neuroglancer's DOM (`capture.mjs`) paired with the
one of the same text nearest it in ours (`rsrects.mjs`), and the offset.

Skia draws glyphs at quarter-pixel x positions (`floor(4x + 1/2)`), so a run a
hair off can land in the next quarter and rasterize differently: those are
flagged `QUARTER`. Runs on other rows (y) or off by whole pixels are
layout differences proper.

    python3 positions.py <scenario> [--all] [--element]

`--all` lists every pair, not just those off; `--element` adds
neuroglancer's element (class) holding each run.
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent


def run(script, state, actions, *args):
    env = dict(os.environ, ACTIONS=actions, FULL="1")
    return subprocess.run(["node", str(here / script), str(here / f"states/{state}.json"), *args], env=env, capture_output=True, text=True, check=True).stdout


def ng_runs(state, actions):
    out = []
    for line in run("capture.mjs", state, actions, "body", "80").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        for r in e.get("runs", []):
            out.append((r[0], r[1], r[2], r[3], e["tag"] + ("." + e["cls"].split()[0] if e.get("cls") else "")))
    return out


def rs_runs(state, actions):
    out = []
    for line in run("rsrects.mjs", state, actions).splitlines():
        if line.startswith("["):
            r = json.loads(line)
            out.append((r[0], r[1], r[2], r[3]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("--all", action="store_true")
    p.add_argument("--element", action="store_true")
    args = p.parse_args()
    spec = json.loads((here / "ui.json").read_text())[args.scenario]
    state, actions = spec.get("state", args.scenario), spec.get("actions", "")
    ng, rs = ng_runs(state, actions), rs_runs(state, actions)
    used = set()
    rows = []
    for text, x, y, w, el in ng:
        best = None
        for k, (t, rx, ry, rw) in enumerate(rs):
            if k in used or t.strip() != text.strip():
                continue
            d = abs(rx - x) + abs(ry - y)
            if d < 6 and (best is None or d < best[0]):
                best = (d, k)
        if best is None:
            rows.append((text, x, y, None, None, "MISSING", el))
            continue
        used.add(best[1])
        _, rx, ry, rw = rs[best[1]]
        flag = ""
        quarter = lambda v: math.floor(4 * v + 0.5)
        if quarter(x) != quarter(rx):
            flag = "QUARTER"
        if abs(ry - y) >= 0.5 or abs(rx - x) >= 1:
            flag = "LAYOUT"
        if abs(rw - w) > 0.02:
            flag += " WIDTH"
        rows.append((text, x, y, rx - x, ry - y, flag.strip(), el))
    shown = [r for r in rows if args.all or r[5]]
    for text, x, y, dx, dy, flag, el in shown:
        offs = "" if dx is None else f"dx {dx:+.5f} dy {dy:+.3f}"
        print(f"{flag:13} {text[:28]!r:32} at {x:9.4f},{y:7.2f}  {offs}" + (f"  [{el}]" if args.element else ""))
    counts = {}
    for r in rows:
        counts[r[5] or "ok"] = counts.get(r[5] or "ok", 0) + 1
    print(f"{args.scenario}: {len(rows)} runs: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
