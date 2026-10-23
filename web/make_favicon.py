#!/usr/bin/env python3
"""Render the static favicon PNGs from the same geometry as favicon.svg.

favicon.svg adapts its tile background to the browser theme via a
prefers-color-scheme media query, but Safari and iOS home screens do
not support SVG favicons, so the static PNG fallbacks ship with the
dark (#181a1b) tile baked in -- the light-chrome/unknown-theme tile,
matching the site's dark-by-default pages -- rounded corners cut out as
real transparency. This script re-renders them
deterministically -- pure stdlib: the SVG's triangle geometry is
supersampled into an RGB raster and encoded as PNG by hand -- so a
palette or geometry change is one re-run away, and `--check` fails
loudly when a committed PNG's decoded output no longer matches the
code's (a differing compressed byte stream with identical scanlines is
zlib-build drift, not staleness, and only draws an informational note).

By default the PNGs are written next to favicon.svg in web/site/.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
SITE_DIR = WEB_DIR / "site"

# Canvas and geometry: must stay identical to web/site/favicon.svg
# (a 32-unit tile with 7-unit rounded corners and two congruent
# triangles, green up sitting higher, red down sitting lower).
CANVAS = 32.0
TILE_RADIUS = 7.0
TILE = (0x18, 0x1A, 0x1B)  # baked-in fallback tile (light chrome / unknown)
GREEN = (0x00, 0xBD, 0x00)
RED = (0xBD, 0x00, 0x00)
UP_TRIANGLE = ((8.0, 8.0), (14.75, 19.5), (1.25, 19.5))
DOWN_TRIANGLE = ((23.25, 24.0), (16.5, 12.5), (30.0, 12.5))

# Renders: name -> (size, corner radius). All three are RGBA with the
# rounded corners cut out as real transparency -- apple-touch-icon.png
# included, by explicit user request. Tradeoff on iOS: transparent
# web-clip pixels composite over black and the system squircle mask
# keeps more corner area than the 7/32 (~21.9%) circular pre-cut, so
# dark slivers can appear at the home-screen corners.
OUTPUTS = {
    "favicon-16x16.png": (16, TILE_RADIUS),
    "favicon-32x32.png": (32, TILE_RADIUS),
    "apple-touch-icon.png": (180, TILE_RADIUS),
}

SUPER_SAMPLE = 8  # sub-samples per pixel edge for antialiasing


def _point_in_triangle(
    x: float, y: float, tri: tuple[tuple[float, float], ...]
) -> bool:
    """Cross-product sign test; both triangles are wound the same way."""
    (ax, ay), (bx, by), (cx, cy) = tri
    return (
        (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0
        and (cx - bx) * (y - by) - (cy - by) * (x - bx) >= 0
        and (ax - cx) * (y - cy) - (ay - cy) * (x - cx) >= 0
    )


def _in_rounded_tile(x: float, y: float, radius: float | None) -> bool:
    if radius is None:
        return 0.0 <= x <= CANVAS and 0.0 <= y <= CANVAS
    if not (0.0 <= x <= CANVAS and 0.0 <= y <= CANVAS):
        return False
    # In a corner quadrant the sub-sample belongs to the tile only if it
    # is within `radius` of that corner's arc center.
    if x < radius and y < radius:
        cx, cy = radius, radius
    elif x > CANVAS - radius and y < radius:
        cx, cy = CANVAS - radius, radius
    elif x < radius and y > CANVAS - radius:
        cx, cy = radius, CANVAS - radius
    elif x > CANVAS - radius and y > CANVAS - radius:
        cx, cy = CANVAS - radius, CANVAS - radius
    else:
        return True
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def render(size: int, radius: float | None) -> bytes:
    """Return a PNG-encoded size x size RGBA image of the icon.

    Each output pixel averages SUPER_SAMPLE^2 sub-samples of the 32-unit
    canvas, so edges antialias without any drawing library. Sub-samples
    outside the rounded tile contribute transparency only: their color
    is excluded from the average and their share sets the pixel's alpha,
    so the corners are true transparency on every page background.
    """
    rows: list[bytes] = []
    step = SUPER_SAMPLE
    scale = CANVAS / size
    for py in range(size):
        row = bytearray()
        for px in range(size):
            r = g = b = covered = 0
            for sy in range(step):
                for sx in range(step):
                    x = (px + (sx + 0.5) / step) * scale
                    y = (py + (sy + 0.5) / step) * scale
                    if not _in_rounded_tile(x, y, radius):
                        continue  # outside the tile: straight alpha 0
                    if _point_in_triangle(x, y, DOWN_TRIANGLE):
                        color = RED
                    elif _point_in_triangle(x, y, UP_TRIANGLE):
                        color = GREEN
                    else:
                        color = TILE
                    r += color[0]
                    g += color[1]
                    b += color[2]
                    covered += 1
            if covered:
                n = step * step
                row += bytes(
                    (
                        round(r / covered),
                        round(g / covered),
                        round(b / covered),
                        round(255 * covered / n),
                    )
                )
            else:
                row += bytes((0, 0, 0, 0))
        rows.append(bytes(row))
    return _encode_png(size, size, rows)


def _encode_png(width: int, height: int, rows: list[bytes]) -> bytes:
    """Minimal PNG encoder: 8-bit RGBA, filter 0 per scanline."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _decoded_scanlines(png: bytes) -> bytes:
    """Inflate a PNG's IDAT stream: the raw scanlines as encoded.

    This is the semantic fingerprint used by --check: the compressed
    byte stream of a PNG depends on the zlib build (zlib and zlib-ng
    differ), but the decoded scanlines only change when the pixels do.
    """
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    idat = bytearray()
    while pos + 12 <= len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        if png[pos + 4 : pos + 8] == b"IDAT":
            idat += png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    if not idat:
        raise ValueError("no IDAT chunk")
    return zlib.decompress(bytes(idat))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_favicon.py",
        description="Render the static favicon PNGs from favicon.svg's geometry",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SITE_DIR),
        help="directory to write the PNGs into (default: web/site)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed PNGs match a fresh render instead of writing",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)
    mismatches: list[str] = []
    byte_drift: list[str] = []
    for name, (size, radius) in OUTPUTS.items():
        png = render(size, radius)
        target = output_dir / name
        if args.check:
            if not target.is_file():
                mismatches.append(name)
                continue
            committed = target.read_bytes()
            try:
                decoded_match = _decoded_scanlines(committed) == _decoded_scanlines(png)
            except (ValueError, struct.error, zlib.error):
                decoded_match = False
            if not decoded_match:
                mismatches.append(name)
            elif committed != png:
                # Same pixels, different compressed bytes: zlib build
                # drift, not staleness.
                byte_drift.append(name)
        else:
            target.write_bytes(png)
            print(f"make_favicon: wrote {target}")
    if args.check:
        if mismatches:
            print(
                "make_favicon: stale PNGs (re-run make_favicon.py and commit): "
                + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 1
        if byte_drift:
            print(
                "make_favicon: decoded output matches; byte stream differs "
                "(zlib build): " + ", ".join(byte_drift),
                file=sys.stderr,
            )
        print("make_favicon: all PNGs match a fresh render")
    return 0


if __name__ == "__main__":
    sys.exit(main())
