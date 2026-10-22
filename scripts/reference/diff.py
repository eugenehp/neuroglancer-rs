"""Compares two screenshots: pixels whose RGB differs by more than `tol`,
outside the masked rectangles.

    python3 diff.py a.png b.png [tol] [x0,y0,x1,y1 ...]
"""

import struct
import sys
import zlib


def load(path):
    """RGB(A) pixels of an 8-bit PNG: (width, height, bytes per pixel, data)."""
    d = open(path, "rb").read()
    i, idat = 8, b""
    while i < len(d):
        n = struct.unpack(">I", d[i : i + 4])[0]
        t, c = d[i + 4 : i + 8], d[i + 8 : i + 8 + n]
        if t == b"IHDR":
            w, h, _, ct = struct.unpack(">IIBB", c[:10])
        if t == b"IDAT":
            idat += c
        i += 12 + n
    raw = zlib.decompress(idat)
    bpp = 4 if ct == 6 else 3
    stride, out, prev, p = w * bpp, bytearray(), bytearray(w * bpp), 0
    for _ in range(h):
        f, line = raw[p], bytearray(raw[p + 1 : p + 1 + stride])
        p += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out += line
        prev = line
    return w, h, bpp, out


def diff(a_path, b_path, tol=1, masks=()):
    w, h, ba, a = load(a_path)
    _, _, bb, b = load(b_path)
    bad = []
    for y in range(h):
        for x in range(w):
            if any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in masks):
                continue
            i, j = (y * w + x) * ba, (y * w + x) * bb
            d = max(abs(a[i + k] - b[j + k]) for k in range(3))
            if d > tol:
                bad.append((x, y, d, tuple(a[i : i + 3]), tuple(b[j : j + 3])))
    return w * h, bad


if __name__ == "__main__":
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    masks = [tuple(map(int, m.split(","))) for m in sys.argv[4:]]
    total, bad = diff(sys.argv[1], sys.argv[2], tol, masks)
    print(f"{len(bad)} of {total} pixels differ by more than {tol}")
    for p in bad[:8]:
        print("  ", p)
