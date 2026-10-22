"""Converts neuroglancer's datasource metadata snapshots
(../neuroglancer/tests/datasource/metadata_snapshots/**/*.snapshot, vitest
pretty-format) to JSON fixtures for crates/ng-datasource/tests.

    python3 scripts/golden/snapshots.py

Typed arrays become arrays, `undefined` and sparse-array holes become null,
±Infinity and NaN become the strings "Infinity", "-Infinity", "NaN".
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT.parent / "neuroglancer/tests/datasource/metadata_snapshots"
OUT = ROOT / "crates/ng-datasource/tests/fixtures/snapshots"


def convert(text: str):
    text = re.sub(r"\b(?:Float32|Float64|Uint8|Uint16|Uint32|Int32|BigUint64)Array \[", "[", text)
    text = re.sub(r"\bundefined\b", "null", text)
    # Holes in sparse arrays: a line holding only a comma.
    text = re.sub(r"(?m)^(\s*),$", r"\1null,", text)
    # BigInt literals (`123n`).
    text = re.sub(r"\b(\d+)n\b", r'"\1"', text)
    # Trailing commas.
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return json.loads(text)


def finite_json(x):
    """JSON has no Infinity/NaN: write them as the strings JS would print."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return "NaN" if x != x else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, list):
        return [finite_json(v) for v in x]
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    return x


OUT.mkdir(parents=True, exist_ok=True)
for path in sorted(SRC.glob("*/*.snapshot")):
    data = finite_json(convert(path.read_text()))
    name = f"{path.parent.name}__{path.stem}.json"
    (OUT / name).write_text(json.dumps(data, indent=1) + "\n")
    print(name)
