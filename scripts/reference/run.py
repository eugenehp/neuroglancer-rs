"""Renders every state in states/ in neuroglancer and neuroglancer-rs
(`compare.mjs`) and checks the screenshots against `states/index.json`: per
state, the RGB tolerance, rectangles to skip (the 3D panel, still a stand-in)
and how many pixels may differ (ties: a voxel boundary or line edge within
float noise of a pixel centre).

    python3 run.py [out-dir] [state ...]

WEBGL=1 runs neuroglancer-rs on its WebGL2 fallback (no WebGPU).

UI=1 compares whole pages instead, both UIs showing, at 1200×800: each
scenario in `ui.json` is a state and pointer/keyboard input
(`compare.mjs`'s ACTIONS), with the pixels allowed to differ by more than 8
levels.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from diff import diff

here = Path(__file__).resolve().parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ng-reference")
out.mkdir(parents=True, exist_ok=True)
ui = bool(os.environ.get("UI"))
index = json.loads((here / ("ui.json" if ui else "states/index.json")).read_text())
names = sys.argv[2:] or list(index)
failed = []
for name in names:
    spec = index[name]
    prefix = out / name
    modes = (["webgl"] if os.environ.get("WEBGL") else []) + (["ui"] if ui else [])
    args = ["node", str(here / "compare.mjs"), str(here / f"states/{spec.get('state', name)}.json"), str(prefix), ",".join(modes)]
    if ui:
        args.append("1200x800")
    env = dict(os.environ, ACTIONS=spec.get("actions", ""))
    if spec.get("ready"):
        env["READY"] = "1"
    subprocess.run(args, check=True, env=env)
    tolerance = spec.get("tolerance", 8 if ui else 1)
    total, bad = diff(f"{prefix}-ng.png", f"{prefix}-rs.png", tolerance, [tuple(m) for m in spec.get("mask", [])])
    ok = len(bad) <= spec.get("allowed", 0)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {len(bad)} of {total} pixels differ (allowed {spec.get('allowed', 0)})")
    if not ok:
        failed.append(name)
        # What the differences are (analyze.py; its report beside the shots).
        subprocess.run([sys.executable, str(here / "analyze.py"), name, "--dir", str(out), "--tol", str(tolerance), "--top", "5"], check=False)
sys.exit(1 if failed else 0)
