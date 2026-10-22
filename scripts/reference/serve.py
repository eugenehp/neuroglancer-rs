"""The reference pages' static server: Python's `http.server` with single
`Range: bytes=` requests answered by 206 Partial Content, as neuroglancer's
byte-range reads (multi-resolution mesh fragments, sharded files) require.

Usage: serve.py PORT (serves the current directory on 127.0.0.1).
"""

import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        match = RANGE.match(header.strip()) if header else None
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = match.groups()
        if first == "":
            # A suffix: the last `last` bytes.
            start, end = max(0, size - int(last or 0)), size - 1
        else:
            start, end = int(first), (int(last) if last else size - 1)
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._remaining = None
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
