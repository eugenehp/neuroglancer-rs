"""Explains a page comparison: where neuroglancer (`<name>-ng.png`) and
neuroglancer-rs (`<name>-rs.png`) differ, grouped into regions, each
classified by what would make it match:

- `shift dx,dy`: the rs pixels moved by a whole pixel match (layout off);
- `subpixel`: shifting halves the difference but does not remove it
  (fractional placement, resampling);
- `colour Δ(r,g,b)`: the same offset throughout (a colour or alpha);
- `missing` / `extra`: one side is flat there and the other is not
  (something not drawn, or drawn that should not be);
- `antialias`: small differences hugging edges (glyph or line
  rasterization);
- `content`: none of these.

Each region is also named by the innermost neuroglancer element holding it
when a DOM capture is given (`--dom`, from `capture.mjs … body 60`).

    python3 analyze.py <name> [--dir /tmp/ng-reference] [--tol 8] [--top 12] [--dom file]
    python3 analyze.py --all [--dir …]    # every scenario there, by kind

Writes `<name>-report.png`: neuroglancer, neuroglancer-rs and the
differences with each region boxed and numbered, then each of the top
regions magnified (neuroglancer, rs, |difference| × 4).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

COLORS = [(255, 64, 64), (64, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 255, 255)]


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def regions(diff, tol, gap):
    """Differing pixels (`diff > tol`) joined within `gap` pixels: per
    region, its slice and its differing pixels' mask."""
    bad = diff > tol
    labels, n = ndimage.label(ndimage.binary_dilation(bad, iterations=gap) if gap else bad)
    out = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = (labels[sl] == k) & bad[sl]
        if mask.any():
            out.append((sl, mask))
    out.sort(key=lambda r: -int(r[1].sum()))
    return out


def classify(a, b, sl, mask, tol):
    """What would make the region match (see the module's docstring)."""
    h, w = a.shape[:2]
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    count = int(mask.sum())
    da = a[sl][mask]
    db = b[sl][mask]
    delta = db - da
    # A uniform offset.
    if count >= 4 and np.abs(delta - delta.mean(axis=0)).max() <= max(4, tol // 2):
        m = delta.mean(axis=0).round().astype(int)
        return f"colour Δ({m[0]:+d},{m[1]:+d},{m[2]:+d})"
    # One side flat.
    pa, pb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    flat = lambda p: p.reshape(-1, 3).std(axis=0).max() < 1.5
    if flat(pa) and not flat(pb):
        return "extra"
    if flat(pb) and not flat(pa):
        return "missing"
    # Shifts of rs within ±3 px.
    m = 3
    Y0, Y1, X0, X1 = max(y0 - m, 0), min(y1 + m, h), max(x0 - m, 0), min(x1 + m, w)
    ref = a[Y0:Y1, X0:X1]
    base = (np.abs(b[Y0:Y1, X0:X1] - ref).max(axis=2) > tol).sum()
    best, best_shift = base, (0, 0)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            if (dx, dy) == (0, 0):
                continue
            sy0, sy1, sx0, sx1 = Y0 - dy, Y1 - dy, X0 - dx, X1 - dx
            if sy0 < 0 or sx0 < 0 or sy1 > h or sx1 > w:
                continue
            c = (np.abs(b[sy0:sy1, sx0:sx1] - ref).max(axis=2) > tol).sum()
            if c < best:
                best, best_shift = c, (dx, dy)
    if best_shift != (0, 0) and best <= base * 0.25:
        return f"shift {best_shift[0]:+d},{best_shift[1]:+d}"
    if best_shift != (0, 0) and best <= base * 0.6:
        return f"subpixel (best {best_shift[0]:+d},{best_shift[1]:+d} leaves {best * 100 // max(base, 1)}%)"
    # Small differences along edges of the neuroglancer image.
    grey = a[y0:y1, x0:x1].mean(axis=2)
    edges = np.hypot(ndimage.sobel(grey, 0), ndimage.sobel(grey, 1)) > 60
    near = ndimage.binary_dilation(edges, iterations=1)
    on_edges = (mask & near).sum() / count
    if on_edges > 0.8 and np.abs(delta).max() < 160:
        return "antialias"
    return "content"


def load_dom(path):
    """Element boxes from `capture.mjs` output: (area, rect, label)."""
    boxes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        x, y, w, h = e["rect"]
        if w <= 0 or h <= 0:
            continue
        label = e["tag"] + ("." + e["cls"].split()[0] if e.get("cls") else "")
        text = " ".join(r[0] for r in e.get("runs", []))[:30]
        boxes.append((w * h, (x, y, x + w, y + h), label + (f' "{text}"' if text else "")))
    boxes.sort(key=lambda b: b[0])
    return boxes


def element_at(boxes, x, y):
    for _, (x0, y0, x1, y1), label in boxes:
        if x0 <= x < x1 and y0 <= y < y1:
            return label
    return ""


def summarize(prefix, tol, gap):
    """Differing pixels by kind, and the regions: (count, kinds, rows)."""
    a, b = load(f"{prefix}-ng.png"), load(f"{prefix}-rs.png")
    diff = np.abs(a - b).max(axis=2)
    found = regions(diff, tol, gap)
    kinds, rows = {}, []
    for i, (sl, mask) in enumerate(found):
        kind = classify(a, b, sl, mask, tol)
        n = int(mask.sum())
        kinds[kind.split(" ")[0]] = kinds.get(kind.split(" ")[0], 0) + n
        rows.append((i + 1, n, (sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start), kind))
    return int((diff > tol).sum()), kinds, rows


def all_scenarios(directory, tol, gap):
    """Every scenario in `directory`: its differences by kind, and the
    totals."""
    totals = {}
    for ng in sorted(Path(directory).glob("*-ng.png")):
        name = ng.name[: -len("-ng.png")]
        if not (ng.parent / f"{name}-rs.png").exists():
            continue
        count, kinds, rows = summarize(ng.parent / name, tol, gap)
        for k, v in kinds.items():
            totals[k] = totals.get(k, 0) + v
        if count:
            biggest = rows[0]
            print(f"{name:40} {count:>7}  " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + f"   largest: {biggest[1]} px {biggest[3]} at {biggest[2][0]},{biggest[2][1]}")
    print("total by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dir", default="/tmp/ng-reference")
    p.add_argument("--tol", type=int, default=8)
    p.add_argument("--gap", type=int, default=2, help="join differences this close")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--dom", help="capture.mjs output to name regions by")
    p.add_argument("--quiet", action="store_true", help="the summary only")
    args = p.parse_args()
    if args.all:
        return all_scenarios(args.dir, args.tol, args.gap)
    prefix = Path(args.dir) / args.name
    a, b = load(f"{prefix}-ng.png"), load(f"{prefix}-rs.png")
    diff = np.abs(a - b).max(axis=2)
    found = regions(diff, args.tol, args.gap)
    total = int((diff > args.tol).sum())
    boxes = load_dom(args.dom) if args.dom else []
    kinds = {}
    rows = []
    for i, (sl, mask) in enumerate(found):
        kind = classify(a, b, sl, mask, args.tol)
        n = int(mask.sum())
        key = kind.split(" ")[0]
        kinds[key] = kinds.get(key, 0) + n
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        where = element_at(boxes, (x0 + x1) / 2, (y0 + y1) / 2) if boxes else ""
        rows.append((i + 1, n, (x0, y0, x1 - x0, y1 - y0), int(diff[sl][mask].max()), kind, where))
    print(f"{args.name}: {total} pixels differ by more than {args.tol} in {len(found)} regions")
    print("  by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if not args.quiet:
        for r in rows[: args.top * 2]:
            print(f"  #{r[0]:<3} {r[1]:>6} px  at {r[2][0]},{r[2][1]} {r[2][2]}×{r[2][3]}  max {r[3]:>3}  {r[4]}" + (f"  [{r[5]}]" if r[5] else ""))
    report(a, b, diff, found, rows, args, prefix)


def report(a, b, diff, found, rows, args, prefix):
    h, w = a.shape[:2]
    heat = np.zeros_like(a)
    heat[..., 0] = np.clip(diff * 4, 0, 255)
    heat[..., 1] = (a.mean(axis=2) * 0.25).astype(np.int16)
    heat[..., 2] = heat[..., 1]
    top = [Image.fromarray(x.astype(np.uint8)) for x in (a, b, heat)]
    scale = 2
    crops = []
    for (sl, _), r in zip(found[: args.top], rows):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        pad = 6
        Y0, Y1, X0, X1 = max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)
        z = max(1, min(8, 160 // max(Y1 - Y0, X1 - X0, 1)))
        parts = [a[Y0:Y1, X0:X1], b[Y0:Y1, X0:X1], np.repeat(np.clip(diff[Y0:Y1, X0:X1] * 4, 0, 255)[..., None], 3, axis=2)]
        parts = [Image.fromarray(p.astype(np.uint8)).resize(((X1 - X0) * z, (Y1 - Y0) * z), Image.NEAREST) for p in parts]
        crops.append((r, parts))
    crop_h = sum(max(p[1][0].height, 14) + 18 for p in crops)
    canvas = Image.new("RGB", (3 * w // scale + 8, h // scale + 4 + crop_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for k, img in enumerate(top):
        canvas.paste(img.resize((w // scale, h // scale)), (k * (w // scale + 4), 0))
    for (sl, _), r in zip(found[: args.top], rows):
        c = COLORS[(r[0] - 1) % len(COLORS)]
        x0, y0, x1, y1 = sl[1].start, sl[0].start, sl[1].stop, sl[0].stop
        for k in range(3):
            ox = k * (w // scale + 4)
            draw.rectangle([ox + x0 // scale - 2, y0 // scale - 2, ox + x1 // scale + 1, y1 // scale + 1], outline=c)
            draw.text((ox + x0 // scale - 2, max(y0 // scale - 12, 0)), str(r[0]), fill=c)
    y = h // scale + 4
    for r, parts in crops:
        c = COLORS[(r[0] - 1) % len(COLORS)]
        draw.text((4, y), f"#{r[0]} {r[1]} px at {r[2][0]},{r[2][1]}: {r[4]}" + (f" [{r[5]}]" if r[5] else ""), fill=c)
        x = 4
        for part in parts:
            canvas.paste(part, (x, y + 14))
            x += part.width + 6
        y += max(parts[0].height, 14) + 18
    out = f"{prefix}-report.png"
    canvas.save(out)
    print(f"  report: {out}")


if __name__ == "__main__":
    main()
