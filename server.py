#!/usr/bin/env python3
"""Minimal HTTP front end: post an image, get SVG back.

    POST /vectorize   body: raw image bytes
                      query: engine=v2 (default) with tolerance, max_colors
                             engine=v1 with mode, colors, grays, threshold, ...
                      -> image/svg+xml

Deliberately stdlib-only and single-threaded: this exists to pin down
the client/server boundary, not to serve traffic.
"""

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from vectorlab import VectorizeOptions, VectorizeRejected, vectorize_image
from vectorlab.v1 import MODE_FULL_COLOR

MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def _number(query, key, default, cast):
    values = query.get(key)
    if not values:
        return default
    try:
        return cast(values[0])
    except (TypeError, ValueError):
        return default


def _trace(payload, query):
    if query.get("engine", ["v2"])[0] == "v1":
        return vectorize_image(io.BytesIO(payload), _options_from_query(query))

    from vectorlab.trace import V2Options, vectorize_v2

    options = V2Options(
        tolerance=_number(query, "tolerance", 0.6, float),
        max_colors=_number(query, "max_colors", 32, int),
        use_primitives=query.get("primitives", ["1"])[0] != "0",
    )
    return vectorize_v2(io.BytesIO(payload), options)


def _options_from_query(query):
    def number(key, default, cast):
        return _number(query, key, default, cast)

    return VectorizeOptions(
        mode=query.get("mode", [MODE_FULL_COLOR])[0],
        color_count=number("colors", 8, int),
        gray_count=number("grays", 5, int),
        threshold=number("threshold", 128, int),
        max_dimension=number("max_dim", 640, int),
        upscale=number("upscale", 1.0, float),
        presmooth=number("presmooth", 0, int),
        simplify_epsilon=number("epsilon", 1.6, float),
        corner_threshold_degrees=number("corner", 42.0, float),
        corner_window=number("corner_window", 2, int),
        smoothness=number("smoothness", 1.0, float),
        min_region_area=number("min_area", 12, int),
        fit_curves=query.get("fit", ["1"])[0] != "0",
        drop_largest_region=query.get("drop_background", ["0"])[0] == "1",
    )


class VectorizeHandler(BaseHTTPRequestHandler):
    server_version = "vectorlab/0.1"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/vectorize":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json(400, {"error": "empty body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "image too large"})
            return

        payload = self.rfile.read(length)

        try:
            result = _trace(payload, parse_qs(parsed.query))
        except VectorizeRejected as rejected:
            self._send_json(
                422,
                {
                    "error": "not suitable for tracing",
                    "reason": rejected.reason,
                    "regions": rejected.region_count,
                },
            )
            return
        except Exception as error:  # noqa: BLE001 - surface any decode/trace failure
            self._send_json(400, {"error": str(error)})
            return

        body = result.svg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Shape-Count", str(result.shape_count))
        self.send_header("X-Anchor-Count", str(result.path_segment_count))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def main(host="127.0.0.1", port=8787):
    server = ThreadingHTTPServer((host, port), VectorizeHandler)
    print(f"vectorlab listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
