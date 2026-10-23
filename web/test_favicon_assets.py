"""Tests for the theme-adaptive favicon assets and their wiring.

favicon.svg is the real icon: an SVG with a prefers-color-scheme media
query that flips the tile background -- cream (#f6f3ec) on dark chrome,
with the dark tile (#181a1b) as the base rule so light chrome and
no-preference browsers match the site's dark-by-default pages. The PNGs
are static fallbacks for browsers that cannot render SVG favicons
(Safari, iOS home screens) with the dark tile baked in, and are
reproducible from web/make_favicon.py (decoded scanlines match a fresh
render), so the check tests re-render and compare.
"""

import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
SITE_DIR = WEB_DIR / "site"

PNG_FILES = {
    "favicon-16x16.png": (16, 16),
    "favicon-32x32.png": (32, 32),
    "apple-touch-icon.png": (180, 180),
}

DARK_TILE = (0x18, 0x1A, 0x1B)  # baked-in fallback tile (light chrome / unknown)
GREEN = (0x00, 0xBD, 0x00)
RED = (0xBD, 0x00, 0x00)
TRANSPARENT = (0, 0, 0, 0)


def test_favicon_svg_is_theme_adaptive():
    svg = (SITE_DIR / "favicon.svg").read_text()
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.get("width") == "32" and root.get("height") == "32"
    # The styled element must be the tile rect itself.
    rects = [el for el in root.iter() if el.tag.endswith("rect")]
    assert any(rect.get("class") == "tile" for rect in rects)
    # Triangle colors are theme-independent and match make_favicon.py.
    assert "#00BD00" in svg
    assert "#BD0000" in svg
    # Pin the mapping DIRECTION, not just the presence of the strings:
    # the dark-chrome media block sets the cream tile, the base rule sets
    # the dark tile. Swapping the two fills must fail this test.
    style = next(el for el in root.iter() if el.tag.endswith("style")).text or ""
    media = re.search(
        r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
        style,
        re.S,
    )
    assert media, "no prefers-color-scheme: dark media block"
    dark_block = media.group(1)
    assert re.search(r"\.tile\s*\{\s*fill:\s*#f6f3ec\s*;?\s*\}", dark_block)
    assert "#181a1b" not in dark_block
    base = style[: media.start()] + style[media.end() :]
    assert re.search(r"\.tile\s*\{\s*fill:\s*#181a1b\s*;?\s*\}", base)
    assert "#f6f3ec" not in base


def test_index_html_links_all_icon_files():
    html = (SITE_DIR / "index.html").read_text()
    assert '<link rel="icon" href="favicon.svg" type="image/svg+xml"' in html
    assert 'href="favicon-32x32.png"' in html
    assert 'href="favicon-16x16.png"' in html
    assert '<link rel="apple-touch-icon" href="apple-touch-icon.png"' in html


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _decoded_scanlines(data: bytes) -> bytes:
    """Inflate a PNG's IDAT stream: the raw scanlines as encoded.

    Comparing these (instead of the compressed bytes) is the semantic
    equality check: a different zlib build may compress the very same
    scanlines to different bytes without any content change.
    """
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    idat = b""
    pos = 8
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        if data[pos + 4 : pos + 8] == b"IDAT":
            idat += data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    return zlib.decompress(idat)


def _decode_pixels(path: Path) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Decode a non-interlaced 8-bit RGBA PNG to (width, height, pixels)."""
    data = path.read_bytes()
    width, height = _png_size(path)
    # IHDR bytes 8-9: bit depth 8, color type 6 (RGBA).
    assert data[24:26] == b"\x08\x06", path.name
    raw = _decoded_scanlines(data)
    stride = 1 + width * 4
    assert len(raw) == stride * height, path.name
    pixels: list[tuple[int, int, int, int]] = []
    for y in range(height):
        # make_favicon.py emits filter 0 (None) on every scanline.
        assert raw[y * stride] == 0, f"{path.name} row {y}: unexpected filter"
        row = raw[y * stride + 1 : (y + 1) * stride]
        pixels.extend(tuple(row[x * 4 : x * 4 + 4]) for x in range(width))
    return width, height, pixels


def test_static_pngs_have_expected_dimensions():
    for name, (width, height) in PNG_FILES.items():
        assert _png_size(SITE_DIR / name) == (width, height), name
        # IHDR: bit depth 8, color type 6 (RGBA): the rounded corners must
        # be transparent, not painted in an opaque color that shows on any
        # page bg.
        assert (SITE_DIR / name).read_bytes()[24:26] == b"\x08\x06", name


def test_static_pngs_decode_dark_tile_with_transparent_corners():
    """Each PNG decodes; all four corners are transparent, tile is dark."""
    for name, (width, height) in PNG_FILES.items():
        w, h, pixels = _decode_pixels(SITE_DIR / name)
        assert (w, h) == (width, height), name

        def pixel(x: int, y: int) -> tuple[int, int, int, int]:
            return pixels[y * w + x]

        # Every rounded corner is cut out as real transparency, on all
        # three files (apple-touch-icon included, by explicit request).
        for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            assert pixel(cx, cy) == TRANSPARENT, (name, cx, cy)
        # Top-middle pixel: inside the tile, opaque dark tile color.
        assert pixel(w // 2, 0)[3] == 255, name
        assert pixel(w // 2, 0)[:3] == DARK_TILE, name


def test_static_pngs_contain_the_triangle_mark():
    """The fallbacks must show the mark: green up, red down triangles."""
    for name in PNG_FILES:
        _w, _h, pixels = _decode_pixels(SITE_DIR / name)
        assert any(px == GREEN + (255,) for px in pixels), name
        assert any(px == RED + (255,) for px in pixels), name


def _run_make_favicon(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(WEB_DIR / "make_favicon.py"), *args],
        capture_output=True,
        text=True,
    )


def test_make_favicon_reproduces_committed_pngs(tmp_path):
    """The generator must re-render the committed images exactly.

    Decoded scanlines must match; the compressed byte stream may differ
    (zlib builds differ). If this fails after a geometry or palette
    edit, the committed PNGs are stale: re-run make_favicon.py and
    commit its output.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(WEB_DIR / "make_favicon.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == ""
    for name in PNG_FILES:
        committed = (SITE_DIR / name).read_bytes()
        fresh = (tmp_path / name).read_bytes()
        assert _decoded_scanlines(fresh) == _decoded_scanlines(committed), name


def test_make_favicon_check_mode(tmp_path):
    """--check exits 0 on a fresh render, 1 on wrong/missing content."""
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _run_make_favicon("--output-dir", str(fresh))
    ok = _run_make_favicon("--check", "--output-dir", str(fresh))
    assert ok.returncode == 0, ok.stderr

    # Valid PNGs, wrong content: one file's bytes swapped for another's.
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    _run_make_favicon("--output-dir", str(swapped))
    (swapped / "apple-touch-icon.png").write_bytes(
        (swapped / "favicon-32x32.png").read_bytes()
    )
    bad = _run_make_favicon("--check", "--output-dir", str(swapped))
    assert bad.returncode == 1
    assert "apple-touch-icon.png" in bad.stderr

    # Missing file: the set is stale.
    missing = tmp_path / "missing"
    missing.mkdir()
    _run_make_favicon("--output-dir", str(missing))
    (missing / "favicon-16x16.png").unlink()
    gone = _run_make_favicon("--check", "--output-dir", str(missing))
    assert gone.returncode == 1
    assert "favicon-16x16.png" in gone.stderr

    # --check must not create directories.
    absent = tmp_path / "absent"
    absent_check = _run_make_favicon("--check", "--output-dir", str(absent))
    assert absent_check.returncode == 1
    assert not absent.exists()


def _recompress(data: bytes) -> bytes:
    """Re-encode a PNG's IDAT at another zlib level: same pixels, new bytes."""
    out = bytearray(data[:8])
    pos = 8
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if tag == b"IDAT":
            payload = zlib.compress(zlib.decompress(payload), 1)
        out += struct.pack(">I", len(payload)) + tag + payload
        out += struct.pack(">I", zlib.crc32(tag + payload))
        pos += 12 + length
    return bytes(out)


def test_make_favicon_check_tolerates_zlib_byte_drift(tmp_path):
    """Same decoded scanlines, different compressed bytes is not stale."""
    out = tmp_path / "zlib-drift"
    out.mkdir()
    _run_make_favicon("--output-dir", str(out))
    target = out / "apple-touch-icon.png"
    drifted = _recompress(target.read_bytes())
    assert drifted != target.read_bytes()
    target.write_bytes(drifted)
    res = _run_make_favicon("--check", "--output-dir", str(out))
    assert res.returncode == 0, res.stderr
    assert "byte stream differs" in res.stderr


def test_make_favicon_creates_missing_output_dir(tmp_path):
    out = tmp_path / "new" / "nested"
    res = _run_make_favicon("--output-dir", str(out))
    assert res.returncode == 0, res.stderr
    for name in PNG_FILES:
        assert (out / name).is_file()
