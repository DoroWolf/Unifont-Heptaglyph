from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.timeTools import timestampFromString
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

CELL = 16                      # full-width bitmap cell, pixels
HALF_CELL = CELL // 2          # 8, half-width ASCII cell width
PIXEL = 64                     # font units per bitmap pixel
UPEM = CELL * PIXEL            # 1024
ASCENT = 14 * PIXEL            # 896
DESCENT = -2 * PIXEL           # -128
ADVANCE = CELL * PIXEL         # 1024, full-width advance

ASCII_FIRST = 0x20
ASCII_LAST = 0x7E
FULLWIDTH_FIRST = 0xFF01
FULLWIDTH_LAST = 0xFF5E
IDEOGRAPHIC_SPACE = 0x3000
HALFWIDTH_HEX_NAME = "unifont-halfwidth-ascii-18.0.01.hex"
FULLWIDTH_HEX_NAME = "unifont-fullwidth-ascii-18.0.01.hex"
UNIFONT_VERSION = "18.0.01"

CP_RE = re.compile(r"^(?:U\+|u\+|0x|0X)?([0-9A-Fa-f]{4,6})$")
FAMILY_DEFAULT = "Unifont Heptaglyph"
STYLE_DEFAULT = "Regular"
VERSION_DEFAULT = "1.110"
# Fixed head.created/modified so repeated builds of the same bitmaps are
# byte-for-byte reproducible instead of depending on the wall clock.
BUILD_TIMESTAMP = timestampFromString("Thu Nov  5 00:00:00 2026")



# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class BitmapGlyph:
    cp: int
    source: Path
    bits: List[List[bool]]
    notes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"u{self.cp:04X}"

    @property
    def width(self) -> int:
        """Bitmap width in pixels: 8 for half-width, 16 for full-width."""
        return len(self.bits[0]) if self.bits else 0

    @property
    def advance(self) -> int:
        return self.width * PIXEL

    @property
    def has_ink(self) -> bool:
        return any(any(row) for row in self.bits)

    @property
    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) in font units, y-up.  None for a blank glyph."""
        xs = [x for y in range(CELL) for x in range(self.width) if self.bits[y][x]]
        if not xs:
            return None
        ys = [y for y in range(CELL) for x in range(self.width) if self.bits[y][x]]
        return (
            min(xs) * PIXEL,
            ASCENT - (max(ys) + 1) * PIXEL,
            (max(xs) + 1) * PIXEL,
            ASCENT - min(ys) * PIXEL,
        )

    @property
    def ink_pixels(self) -> int:
        return sum(sum(1 for v in row if v) for row in self.bits)


# --------------------------------------------------------------------------- #
# bitmap loading
# --------------------------------------------------------------------------- #
def parse_codepoint(raw: str) -> int:
    match = CP_RE.match(raw.strip())
    if not match:
        raise ValueError(f"not a code point: {raw!r}")
    value = int(match.group(1), 16)
    if not (0 <= value <= 0x10FFFF):
        raise ValueError(f"code point out of range: {raw!r}")
    return value


def load_bitmap(path: Path) -> BitmapGlyph:
    """Read a PNG as a 16x16 boolean ink matrix.

    Palette/transparency is handled by converting to RGBA.  Any pixel with
    alpha > 127 counts as ink; the source set is strictly black-on-transparent.
    Images larger than 16x16 are reduced with a 16x16 window that contains all
    ink, preserving the glyph's position inside the cell (a 17x17 source keeps
    its 1 px canvas border as inset).  Smaller images are trimmed and placed at
    the top-left.  Every adjustment is recorded in the manifest.
    """
    try:
        with Image.open(path) as raw:
            raw.load()
            im = raw.convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - report the file, keep building
        raise ValueError(f"cannot read {path.name}: {exc}") from exc

    width, height = im.size
    mask = [[im.getpixel((x, y))[3] > 127 for x in range(width)] for y in range(height)]
    notes: List[str] = []

    if (width, height) != (CELL, CELL):
        xs = [x for y in range(height) for x in range(width) if mask[y][x]]
        ys = [y for y in range(height) for x in range(width) if mask[y][x]]
        if not xs:
            raise ValueError(f"{path.name}: {width}x{height} image is empty")
        left, top = min(xs), min(ys)
        right, bottom = max(xs) + 1, max(ys) + 1

        if width > CELL or height > CELL:
            # Take a 16x16 window that still contains all ink instead of
            # cropping to the ink box.  This preserves the glyph's position
            # inside the cell: a 17x17 canvas made by adding a 1 px border
            # around a 16x16 image therefore keeps that border's inset.
            def fits(limit: int, start: int, end: int) -> List[int]:
                last = limit - CELL
                return list(range(max(0, end - CELL), min(start, last) + 1))

            offsets_x = fits(width, left, right)
            offsets_y = fits(height, top, bottom)
            if not offsets_x or not offsets_y:
                raise ValueError(
                    f"{path.name}: ink box is {right - left}x{bottom - top}, "
                    f"which cannot fit a {CELL}x{CELL} window"
                )
            candidates = [(ox, oy) for oy in offsets_y for ox in offsets_x]
            preferred = (1 if width > CELL else 0, 1 if height > CELL else 0)
            if preferred in candidates:
                offset = preferred
            else:
                offset = min(
                    candidates,
                    key=lambda pair: (abs(pair[0] - preferred[0])
                                      + abs(pair[1] - preferred[1]), pair),
                )
            ox, oy = offset
            bits = [row[ox:ox + CELL] for row in mask[oy:oy + CELL]]
            notes.append(
                f"source {width}x{height}; 16x16 window at ({ox},{oy}); "
                f"ink box {right - left}x{bottom - top} at ({left},{top})"
            )
        else:
            # Smaller than 16x16: trim to the ink and place it at the top-left.
            cropped = [row[left:right] for row in mask[top:bottom]]
            bits = [[False] * CELL for _ in range(CELL)]
            for y, row in enumerate(cropped):
                for x, value in enumerate(row):
                    bits[y][x] = value
            notes.append(
                f"source {width}x{height}; ink box {right - left}x{bottom - top} "
                f"at ({left},{top}); placed at 0,0"
            )
    else:
        bits = mask

    # Guard against anti-aliased or coloured inputs: only alpha matters, but a
    # mid-alpha edge would silently turn into a hard edge here.
    alphas = {im.getpixel((x, y))[3] for y in range(height) for x in range(width)}
    if any(0 < a < 255 for a in alphas):
        notes.append("contains semi-transparent pixels; thresholded at alpha > 127")

    return BitmapGlyph(cp=0, source=path, bits=bits, notes=notes)


def collect_bitmaps(src: Path) -> Tuple[List[BitmapGlyph], List[Tuple[Path, str]]]:
    glyphs: Dict[int, BitmapGlyph] = {}
    skipped: List[Tuple[Path, str]] = []
    for path in sorted(src.glob("*.png"), key=lambda p: p.name.casefold()):
        try:
            cp = parse_codepoint(path.stem)
        except ValueError:
            skipped.append((path, "file name is not a code point"))
            continue
        if cp in glyphs:
            skipped.append((path, f"duplicate code point U+{cp:04X} "
                                  f"(kept {glyphs[cp].source.name})"))
            continue
        try:
            glyph = load_bitmap(path)
        except ValueError as exc:
            skipped.append((path, str(exc)))
            continue
        glyph.cp = cp
        glyphs[cp] = glyph
    return [glyphs[cp] for cp in sorted(glyphs)], skipped


# --------------------------------------------------------------------------- #
# .hex
# --------------------------------------------------------------------------- #
def bits_to_hex(bits: Sequence[Sequence[bool]]) -> str:
    width = len(bits[0])
    out: List[str] = []
    for row in bits:
        for x in range(0, width, 4):
            value = (int(row[x]) << 3) | (int(row[x + 1]) << 2) | \
                    (int(row[x + 2]) << 1) | int(row[x + 3])
            out.append(f"{value:X}")
    return "".join(out)


def hex_to_bits(text: str, width: Optional[int] = None) -> List[List[bool]]:
    """Decode one Unifont bitmap; either 8x16 (32 digits) or 16x16 (64)."""
    text = text.strip()
    if width is None:
        width = len(text) // 4
    if width not in (HALF_CELL, CELL):
        raise ValueError(f"unsupported bitmap width {width}; expected 8 or 16")
    expected = CELL * width // 4
    if len(text) != expected:
        raise ValueError(f"expected {expected} hex digits for {width}x16, got {len(text)}")
    bits = [[False] * width for _ in range(CELL)]
    for i, ch in enumerate(text):
        value = int(ch, 16)
        row, col = divmod(i * 4, width)
        for bit in range(4):
            bits[row][col + bit] = bool(value & (8 >> bit))
    return bits


def load_hex_range(path: Path, first: int, last: int, width: int,
                   label: str) -> Dict[int, List[List[bool]]]:
    """Read a Unifont .hex code point range and return its bitmaps."""
    if not path.is_file():
        raise FileNotFoundError(f"Unifont {label} .hex not found: {path}")
    records: Dict[int, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path.name}:{lineno}: expected CODEPOINT:BITMAP")
        raw_cp, raw_bits = line.split(":", 1)
        try:
            cp = int(raw_cp, 16)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{lineno}: bad code point {raw_cp!r}") from exc
        if first <= cp <= last:
            records[cp] = raw_bits.strip()

    missing = [cp for cp in range(first, last + 1) if cp not in records]
    if missing:
        raise ValueError(f"{path.name}: missing {label} code point(s): "
                         + ", ".join(f"U+{cp:04X}" for cp in missing[:12]))

    bitmaps: Dict[int, List[List[bool]]] = {}
    for cp in range(first, last + 1):
        raw_bits = records[cp]
        got_width = len(raw_bits) // 4
        if got_width != width:
            raise ValueError(f"{path.name}: U+{cp:04X} has {got_width} columns; "
                             f"expected {width}")
        bitmaps[cp] = hex_to_bits(raw_bits, width)
    return bitmaps


def load_ascii_glyphs(halfwidth_path: Path,
                      fullwidth_path: Path) -> List[BitmapGlyph]:
    """Load GNU Unifont's native half-width and full-width ASCII glyphs.

    U+0020..U+007E keep their 8x16 half-width bitmaps (8 px advance),
    U+3000 keeps its blank 16x16 full-width space, and U+FF01..U+FF5E keep
    their 16x16 full-width bitmaps (16 px advance).  The sets are separate code
    points, so "ASCII", full-width "ＡＳＣＩＩ", and full-width spaces
    (U+3000) can be mixed at their natural widths instead of remapping one
    block onto the other.
    """
    half = load_hex_range(halfwidth_path, ASCII_FIRST, ASCII_LAST,
                          HALF_CELL, "half-width ASCII")
    space = load_hex_range(fullwidth_path, IDEOGRAPHIC_SPACE, IDEOGRAPHIC_SPACE,
                           CELL, "ideographic space")
    full = load_hex_range(fullwidth_path, FULLWIDTH_FIRST, FULLWIDTH_LAST,
                          CELL, "full-width ASCII")

    glyphs: List[BitmapGlyph] = []
    for cp in range(ASCII_FIRST, ASCII_LAST + 1):
        note = (f"blank half-width U+0020 space (GNU Unifont {UNIFONT_VERSION})"
                if cp == 0x20 else
                f"GNU Unifont {UNIFONT_VERSION} half-width U+{cp:04X}")
        glyphs.append(BitmapGlyph(cp=cp, source=halfwidth_path, bits=half[cp],
                                  notes=[note]))
    glyphs.append(BitmapGlyph(
        cp=IDEOGRAPHIC_SPACE, source=fullwidth_path,
        bits=space[IDEOGRAPHIC_SPACE],
        notes=[f"blank full-width U+3000 space (GNU Unifont {UNIFONT_VERSION})"],
    ))
    for cp in range(FULLWIDTH_FIRST, FULLWIDTH_LAST + 1):
        glyphs.append(BitmapGlyph(
            cp=cp, source=fullwidth_path, bits=full[cp],
            notes=[f"GNU Unifont {UNIFONT_VERSION} full-width U+{cp:04X}"],
        ))
    return glyphs


def write_hex(path: Path, glyphs: Sequence[BitmapGlyph]) -> None:
    lines = [f"{glyph.cp:04X}:{bits_to_hex(glyph.bits)}" for glyph in glyphs]
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


# --------------------------------------------------------------------------- #
# TTF
# --------------------------------------------------------------------------- #
def make_glyph(bits: Sequence[Sequence[bool]]):
    """Return a TrueType glyph of filled rectangles for each horizontal run."""
    pen = TTGlyphPen(None)
    width = len(bits[0])
    for row in range(CELL):
        col = 0
        while col < width:
            if not bits[row][col]:
                col += 1
                continue
            start = col
            while col < width and bits[row][col]:
                col += 1
            x0, x1 = start * PIXEL, col * PIXEL
            y1 = ASCENT - row * PIXEL
            y0 = y1 - PIXEL
            # TrueType outer contours are clockwise in y-up coordinates.
            pen.moveTo((x0, y0))
            pen.lineTo((x0, y1))
            pen.lineTo((x1, y1))
            pen.lineTo((x1, y0))
            pen.closePath()
    return pen.glyph()


def make_notdef():
    bits = [[False] * CELL for _ in range(CELL)]
    for x in range(2, 14):
        bits[2][x] = bits[13][x] = True
    for y in range(2, 14):
        bits[y][2] = bits[y][13] = True
    return make_glyph(bits)


def add_gasp(font: TTFont) -> None:
    gaspr = newTable("gasp")
    gaspr.version = 1
    # 0x000F: grid-fit + grayscale + symmetric smoothing.  The font is
    # unhinted, but this tells rasterisers not to use mono-only rendering.
    gaspr.gaspRange = {0xFFFF: 0x000F}
    font["gasp"] = gaspr


def build_ttf(path: Path, glyphs: Sequence[BitmapGlyph], family: str,
              style: str, version: str) -> Dict[str, Tuple[int, int]]:
    order: List[str] = [".notdef"]
    cmap: Dict[int, str] = {}
    glyf: Dict[str, object] = {".notdef": make_notdef()}
    hmtx: Dict[str, Tuple[int, int]] = {".notdef": (ADVANCE, 2 * PIXEL)}
    vmtx: Dict[str, Tuple[int, int]] = {".notdef": (ADVANCE, ASCENT - 12 * PIXEL)}

    for glyph in glyphs:
        name = glyph.name
        order.append(name)
        cmap[glyph.cp] = name
        glyf[name] = make_glyph(glyph.bits)
        bbox = glyph.bbox
        if bbox is None:
            hmtx[name] = (glyph.advance, 0)
            vmtx[name] = (ADVANCE, 0)
        else:
            x0, _y0, _x1, y1 = bbox
            hmtx[name] = (glyph.advance, x0)
            vmtx[name] = (ADVANCE, ASCENT - y1)

    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(hmtx)
    fb.setupVerticalMetrics(vmtx)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT, lineGap=0,
                             caretSlopeRise=1, caretSlopeRun=0, caretOffset=0)
    fb.setupVerticalHeader(ascent=ASCENT, descent=DESCENT, lineGap=0,
                           caretSlopeRise=0, caretSlopeRun=1, caretOffset=0)
    fb.setupOS2(
        version=4,
        sTypoAscender=ASCENT, sTypoDescender=DESCENT, sTypoLineGap=0,
        usWinAscent=ASCENT, usWinDescent=-DESCENT,
        usWeightClass=400, usWidthClass=5, fsType=0, fsSelection=0x40,
        achVendID="WLDN", sFamilyClass=0,
        sxHeight=8 * PIXEL, sCapHeight=12 * PIXEL,
        yStrikeoutSize=1 * PIXEL, yStrikeoutPosition=4 * PIXEL,
        usFirstCharIndex=0x20, usLastCharIndex=0xFFFF,
        usDefaultChar=0, usBreakChar=0x20, usMaxContext=0,
        ulCodePageRange1=0, ulCodePageRange2=0,
    )
    fb.setupNameTable({
        "familyName": family,
        "styleName": style,
        "uniqueFontIdentifier": f"{version};WLDN;{family.replace(' ', '')}-{style}",
        "fullName": f"{family} {style}",
        "version": f"Version {version}",
        "psName": f"{family.replace(' ', '')}-{style}",
        "typographicFamily": family,
        "typographicSubfamily": style,
        "compatibleFullName": f"{family} {style}",
        "wwsFamilyName": family,
        "wwsSubfamilyName": style,
    })
    # Half-width ASCII advances 512 units, every other glyph 1024, so the
    # face is deliberately not fixed pitch.
    fb.setupPost(keepGlyphNames=True, isFixedPitch=0,
                 underlinePosition=-DESCENT // 2, underlineThickness=PIXEL)

    font = fb.font
    font["hhea"].recalc(font)
    font["vhea"].recalc(font)
    font["maxp"].recalc(font)
    font["OS/2"].recalcAvgCharWidth(font)
    font["OS/2"].recalcUnicodeRanges(font)
    # The PUA-A range has no dedicated OS/2 bit; the recalc above sets
    # "Non-Plane 0" for the f0000..ffffd code points, which is correct.

    # Global bounding box (head) from the actual glyphs.
    boxes = [g for g in glyf.values() if g.numberOfContours > 0]
    if boxes:
        font["head"].xMin = min(g.xMin for g in boxes)
        font["head"].yMin = min(g.yMin for g in boxes)
        font["head"].xMax = max(g.xMax for g in boxes)
        font["head"].yMax = max(g.yMax for g in boxes)
    font["head"].fontRevision = float(version)
    font["head"].created = BUILD_TIMESTAMP
    font["head"].modified = BUILD_TIMESTAMP
    font["head"].flags = 0x000B
    font["head"].lowestRecPPEM = 8
    font["head"].macStyle = 0
    font["OS/2"].fsSelection = 0x40
    font["OS/2"].usFirstCharIndex = 0x20
    font["OS/2"].usLastCharIndex = 0xFFFF

    add_gasp(font)
    font.save(path)
    return {name: hmtx[name] for name in order}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_outputs(hex_path: Path, ttf_path: Path,
                     glyphs: Sequence[BitmapGlyph]) -> List[str]:
    problems: List[str] = []

    # 1. .hex round-trip
    expected = {g.cp: g.bits for g in glyphs}
    seen: Dict[int, List[List[bool]]] = {}
    for lineno, line in enumerate(hex_path.read_text(encoding="ascii").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        try:
            raw_cp, raw_bits = line.split(":", 1)
            cp = int(raw_cp, 16)
            bits = hex_to_bits(raw_bits)
        except ValueError as exc:
            problems.append(f".hex line {lineno}: {exc}")
            continue
        if cp in seen:
            problems.append(f".hex line {lineno}: duplicate U+{cp:04X}")
        seen[cp] = bits
    if set(seen) != set(expected):
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        if missing:
            problems.append(f".hex missing {len(missing)} code point(s): "
                            + ", ".join(f"U+{cp:04X}" for cp in missing[:8]))
        if extra:
            problems.append(f".hex has {len(extra)} unexpected code point(s): "
                            + ", ".join(f"U+{cp:04X}" for cp in extra[:8]))
    for cp, bits in expected.items():
        if seen.get(cp) != bits:
            problems.append(f".hex bitmap mismatch at U+{cp:04X}")

    # 2. TTF structure
    ttf = TTFont(ttf_path)
    cmap = ttf.getBestCmap()
    for cp in expected:
        if cp not in cmap:
            problems.append(f"TTF cmap missing U+{cp:04X}")
    cmap_subtables = [(t.platformID, t.platEncID, t.format) for t in ttf["cmap"].tables]
    if not any(fmt == 12 for _p, _e, fmt in cmap_subtables):
        problems.append("TTF cmap has no format 12 subtable (required for plane 15)")
    if ttf["head"].unitsPerEm != UPEM:
        problems.append(f"TTF unitsPerEm is {ttf['head'].unitsPerEm}, expected {UPEM}")
    if ttf["hhea"].ascent != ASCENT or ttf["hhea"].descent != DESCENT:
        problems.append("TTF hhea metrics do not match the 14/2 px cell")
    if ttf["OS/2"].fsType != 0:
        problems.append("TTF is not installable (OS/2 fsType != 0)")
    for glyph in glyphs:
        advance, _lsb = ttf["hmtx"][glyph.name]
        if advance != glyph.advance:
            problems.append(f"TTF advance for U+{glyph.cp:04X} is {advance}, "
                            f"expected {glyph.advance}")
    if ttf["post"].isFixedPitch != 0:
        problems.append("TTF still claims to be fixed pitch although widths vary")

    # 3. Verify the actual outline rectangles against every source bitmap.
    #    Rasterising through FreeType would be affected by hinting (for example
    #    Unifont's two-row dot on "i" and "j"), so inspect the generated
    #    contours directly instead.
    try:
        ImageFont.truetype(str(ttf_path), CELL)
    except OSError as exc:
        problems.append(f"Pillow cannot load the built TTF: {exc}")
        return problems

    for glyph in glyphs:
        got = [[False] * glyph.width for _ in range(CELL)]
        outline = ttf["glyf"][glyph.name]
        if outline.numberOfContours > 0:
            coordinates = [tuple(point) for point in outline.coordinates]
            start = 0
            for end in outline.endPtsOfContours:
                points = coordinates[start:end + 1]
                start = end + 1
                if len(points) != 4:
                    problems.append(f"U+{glyph.cp:04X}: contour has {len(points)} points")
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                if len(set(xs)) != 2 or len(set(ys)) != 2:
                    problems.append(f"U+{glyph.cp:04X}: contour is not axis-aligned")
                    continue
                area2 = sum(
                    points[i][0] * points[(i + 1) % 4][1]
                    - points[(i + 1) % 4][0] * points[i][1]
                    for i in range(4)
                )
                if area2 >= 0:
                    problems.append(f"U+{glyph.cp:04X}: contour is not clockwise")
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                if (x0 % PIXEL or x1 % PIXEL or y0 % PIXEL or y1 % PIXEL
                        or x0 < 0 or x1 > glyph.advance
                        or y0 < DESCENT or y1 > ASCENT):
                    problems.append(f"U+{glyph.cp:04X}: rectangle is off the pixel grid")
                    continue
                row_start = (ASCENT - y1) // PIXEL
                row_end = (ASCENT - y0) // PIXEL - 1
                for row in range(row_start, row_end + 1):
                    for col in range(x0 // PIXEL, x1 // PIXEL):
                        got[row][col] = True

        if got != glyph.bits:
            diff = sum(1 for y in range(CELL) for x in range(glyph.width)
                       if got[y][x] != glyph.bits[y][x])
            problems.append(
                f"TTF outline mismatch at U+{glyph.cp:04X} ({glyph.source.name}): "
                f"{diff} pixel(s)"
            )
            if len(problems) > 30:
                problems.append("... more outline mismatches omitted")
                break

    return problems


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def replace_with_retry(source: Path, target: Path, attempts: int = 6,
                       delay: float = 0.25) -> None:
    """Atomically replace *target*, retrying while a viewer briefly holds it."""
    last_error: Optional[OSError] = None
    for _attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:  # Windows can return 13 or 22 for a busy file
            last_error = exc
            time.sleep(delay)
    raise last_error if last_error is not None else OSError(f"cannot replace {target}")


def load_label_font(size: int, bold: bool = False):
    names = ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf"] if bold else []
    names += ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_glyph_cell(ttf_path: Path, cp: int, size: int = CELL) -> Image.Image:
    """Render one code point at *size* px and return an L-mode cell."""
    font = ImageFont.truetype(str(ttf_path), size)
    baseline = ASCENT * size // UPEM
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, baseline), chr(cp), font=font, fill=0, anchor="ls")
    # The master is 1-bit; drop the rasteriser's edge grey so the sheet shows
    # the actual pixel grid instead of anti-aliasing.
    return image.point(lambda value: 0 if value < 128 else 255)


def render_contact_sheet(ttf_path: Path, glyphs: Sequence[BitmapGlyph], out_path: Path,
                         scale: int = 3, content_width: int = 1120) -> None:
    """Draw every glyph at *scale*, sizing cells by the glyph's own advance."""
    pad = 5
    gap = 8
    margin = 16
    header_h = 58
    label_h = 15
    cell_h = CELL * scale

    def slot_width(glyph: BitmapGlyph) -> int:
        return glyph.width * scale + 2 * pad + gap

    # Flow the glyphs into rows: an 8 px half-width cell takes half the room
    # of a 16 px full-width cell, matching the advances in the font.
    rows: List[List[BitmapGlyph]] = []
    current: List[BitmapGlyph] = []
    current_width = 0
    for glyph in glyphs:
        if current and current_width + slot_width(glyph) > content_width:
            rows.append(current)
            current = []
            current_width = 0
        current.append(glyph)
        current_width += slot_width(glyph)
    if current:
        rows.append(current)

    sheet_width = max(sum(slot_width(g) for g in row) for row in rows) + margin * 2
    sheet_height = header_h + len(rows) * (cell_h + label_h) + margin
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = load_label_font(17, bold=True)
    label_font = load_label_font(8)

    draw.text((margin, margin), "Unifont Heptaglyph", fill=(20, 20, 20), font=title_font)
    draw.text((margin, margin + 24),
              f"{len(glyphs)} glyphs: half/full-width ASCII + U+3000 + Xingyu PUA",
              fill=(120, 120, 120), font=label_font)

    y = header_h
    for row in rows:
        x = margin
        for glyph in row:
            cell_px = glyph.width * scale
            bbox = glyph.bbox
            if bbox is not None:
                cell = render_glyph_cell(ttf_path, glyph.cp)
                cell = cell.resize((CELL * scale, CELL * scale), Image.NEAREST)
                cell = cell.crop((0, 0, cell_px, cell_h))
                sheet.paste(Image.merge("RGB", (cell, cell, cell)), (x + pad, y))
            draw.rectangle([x + pad, y, x + pad + cell_px - 1, y + cell_h - 1],
                           outline=(228, 228, 228))
            draw.text((x + pad + 1, y + cell_h + 1), f"{glyph.cp:04X}",
                      fill=(125, 125, 125), font=label_font)
            x += slot_width(glyph)
        y += cell_h + label_h

    temp_path = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    try:
        sheet.save(temp_path, optimize=True)
        replace_with_retry(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_preview_html(path: Path, family: str, glyphs: Sequence[BitmapGlyph],
                       font_path: Path) -> None:
    """A self-contained page that embeds the font and lists every glyph."""
    import base64

    suffix = font_path.suffix.lower()
    mime, fmt = ("font/woff2", "woff2") if suffix == ".woff2" else ("font/ttf", "truetype")
    payload = base64.b64encode(font_path.read_bytes()).decode("ascii")
    pua = [g for g in glyphs if g.cp > 0xFFFF]
    step = max(1, len(pua) // 10)
    pua_sample = "".join(chr(g.cp) for g in pua[::step][:10])
    fullwidth_sample = "".join(
        chr(IDEOGRAPHIC_SPACE) if ch == " " else
        chr(FULLWIDTH_FIRST + (ord(ch) - 0x21)) if 0x21 <= ord(ch) <= ASCII_LAST
        else ch
        for ch in "The quick brown fox 0123456789."
    )
    sample = ("Half-width: The quick brown fox 0123456789.  "
              f"Full-width: {fullwidth_sample}  " + pua_sample)
    cells = "\n".join(
        f'<div class="cell"><span class="glyph">{chr(g.cp)}</span>'
        f'<span class="cp">{g.cp:04X}</span></div>'
        for g in glyphs
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>{family} - preview</title>
<style>
  :root {{ color-scheme: light; }}
  @font-face {{
    font-family: '{family}';
    src: url(data:{mime};base64,{payload}) format('{fmt}');
    font-weight: 400;
    font-style: normal;
    font-synthesis: none;
  }}
  body {{ margin: 0; padding: 32px; background: #fff; color: #171717;
         font: 14px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif; }}
  h1 {{ font-size: 28px; margin: 0 0 2px; }}
  .sub {{ color: #666; margin: 0 0 20px; }}
  .sample {{ font-family: '{family}'; font-size: 32px; line-height: 1.1;
             border: 1px solid #ddd; padding: 12px; margin: 16px 0 24px;
             min-height: 56px; }}
  .sample:focus {{ outline: 2px solid #999; outline-offset: 1px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
           gap: 3px; }}
  .cell {{ border: 1px solid #ececec; padding: 3px 0 2px; text-align: center; }}
  .glyph {{ display: block; font-family: '{family}'; font-size: 16px;
            line-height: 16px; height: 16px; }}
  .cp {{ display: block; font: 8px/1 ui-monospace, Consolas, monospace;
         color: #999; margin-top: 2px; }}
  footer {{ margin-top: 24px; color: #888; font-size: 12px; }}
</style>
<h1>{family}</h1>
<p class="sub">{len(glyphs)} glyphs &middot; half/full-width ASCII + U+3000 + Xingyu PUA &middot;
U+{glyphs[0].cp:04X}-U+{glyphs[-1].cp:04X} &middot; font embedded, no server needed</p>
<p>点击下方文本可编辑，用来测试字体：</p>
<div class="sample" contenteditable="true">{sample}</div>
<div class="grid">
{cells}
</div>
</html>
"""
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        temp_path.write_text(document, encoding="utf-8", newline="\n")
        replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_manifest(path: Path, glyphs: Sequence[BitmapGlyph]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["codepoint", "character", "glyph_name", "width",
                         "source", "ink_pixels", "notes"])
        for glyph in glyphs:
            writer.writerow([
                f"U+{glyph.cp:04X}", chr(glyph.cp), glyph.name, glyph.width,
                glyph.source.name, glyph.ink_pixels,
                "; ".join(glyph.notes),
            ])


def load_table_codepoints(map_path: Path) -> Dict[int, str]:
    """Read xingyu.json and return {code point: name} for coverage reporting."""
    document = json.loads(map_path.read_text(encoding="utf-8"))
    entries = document.get("entries", document)
    result: Dict[int, str] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("cp")
            if raw is None:
                continue
            try:
                cp = parse_codepoint(str(raw))
            except ValueError:
                continue
            result.setdefault(cp, str(entry.get("name", "")))
    return result

def write_missing(path: Path, missing: Iterable[Tuple[int, str]]) -> None:
    lines = ["# xingyu.json code points with no bitmap in glyphs_cp",
             "# format: CODEPOINT<TAB>name"]
    for cp, name in missing:
        lines.append(f"{cp:04X}\t{name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_zip(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, f"{out_dir.name}/{path.relative_to(out_dir)}")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, Path.cwd()):
        if (candidate / "xingyu.json").is_file() or (candidate / "glyphs_cp").is_dir():
            return candidate
    return Path.cwd()


def main() -> int:
    root = project_root()
    src = root / "glyphs_cp"
    out_dir = root / "dist" / "UnifontHeptaglyph"
    map_path = root / "xingyu.json"
    halfwidth_path = root / "data" / HALFWIDTH_HEX_NAME
    fullwidth_path = root / "data" / FULLWIDTH_HEX_NAME

    if not src.is_dir():
        print(f"error: source folder not found: {src}", file=sys.stderr)
        return 1

    pua_glyphs, skipped = collect_bitmaps(src)
    if not pua_glyphs:
        print(f"error: no code-point-named PNGs found in {src}", file=sys.stderr)
        return 1

    try:
        ascii_glyphs = load_ascii_glyphs(halfwidth_path, fullwidth_path)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load GNU Unifont ASCII glyphs: {exc}", file=sys.stderr)
        return 1

    glyphs = sorted(ascii_glyphs + pua_glyphs, key=lambda glyph: glyph.cp)
    if not glyphs:
        print("error: no glyphs to build", file=sys.stderr)
        return 1
    duplicates = len(glyphs) - len({glyph.cp for glyph in glyphs})
    if duplicates:
        print(f"error: {duplicates} duplicate code point(s) between ASCII and PUA sets",
              file=sys.stderr)
        return 1

    half_count = sum(1 for glyph in ascii_glyphs if glyph.width == HALF_CELL)
    full_count = sum(1 for glyph in ascii_glyphs if glyph.width == CELL)

    print(f"ascii  : {halfwidth_path.name} + {fullwidth_path.name}  "
          f"({len(ascii_glyphs)} glyphs: {half_count} half-width, "
          f"{full_count} full-width incl. U+3000)")
    print(f"source : {src}  ({len(pua_glyphs)} bitmaps)")
    if skipped:
        print(f"skipped: {len(skipped)} file(s)")
        for path, reason in skipped:
            print(f"  ! {path.name}: {reason}")

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {".ttf", ".hex", ".woff2", ".csv", ".md", ".txt", ".zip",
                 ".png", ".html"}
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in generated:
            path.unlink()

    stem = FAMILY_DEFAULT.replace(" ", "")
    ttf_path = out_dir / f"{stem}.ttf"
    hex_path = out_dir / f"{stem}.hex"
    woff2_path = out_dir / f"{stem}.woff2"
    preview_path = out_dir / f"{stem}-preview.png"
    preview_html_path = out_dir / f"{stem}-preview.html"

    print("building .hex ...")
    write_hex(hex_path, glyphs)

    print("building .ttf ...")
    build_ttf(ttf_path, glyphs, FAMILY_DEFAULT, STYLE_DEFAULT, VERSION_DEFAULT)

    print("validating ...")
    problems = validate_outputs(hex_path, ttf_path, glyphs)
    if problems:
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        print(f"error: {len(problems)} validation problem(s); fonts were written but "
              "should be inspected", file=sys.stderr)
        return 2

    woff2_ok = False
    try:
        font = TTFont(ttf_path)
        font.flavor = "woff2"
        font.save(woff2_path)
        font.close()
        woff2_ok = True
        print(f"woff2  : {woff2_path.name}")
    except Exception as exc:  # noqa: BLE001 - WOFF2 is optional
        print(f"  ! could not build WOFF2 ({exc}); TTF/HEX are still valid",
              file=sys.stderr)

    samples_ok = True
    print("updating sample PNG/HTML ...")
    try:
        render_contact_sheet(ttf_path, glyphs, preview_path)
        # Prefer the small WOFF2, but fall back to the TTF so the page is
        # still updated if WOFF2 compression is unavailable.
        html_font = woff2_path if woff2_ok and woff2_path.exists() else ttf_path
        write_preview_html(preview_html_path, FAMILY_DEFAULT, glyphs, html_font)
        print(f"samples : {preview_path.name}, {preview_html_path.name}")
    except Exception as exc:  # noqa: BLE001 - keep the font build usable
        samples_ok = False
        print(f"  ! could not update samples ({exc})", file=sys.stderr)

    table: Optional[Dict[int, str]] = None
    if map_path.is_file():
        try:
            table = load_table_codepoints(map_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ! could not read {map_path.name} for coverage: {exc}", file=sys.stderr)
    if table is not None:
        missing = [(cp, table[cp]) for cp in sorted(set(table) - {g.cp for g in pua_glyphs})]
        if missing:
            write_missing(out_dir / "missing-codepoints.txt", missing)
        else:
            (out_dir / "missing-codepoints.txt").unlink(missing_ok=True)

    write_manifest(out_dir / "glyph-manifest.csv", glyphs)
    for source_file in (halfwidth_path, fullwidth_path):
        shutil.copy2(source_file, out_dir / source_file.name)
    for name in ("UNIFONT-LICENSE.txt", "UNIFONT-NOTICE.txt"):
        source_file = halfwidth_path.parent / name
        if source_file.is_file():
            shutil.copy2(source_file, out_dir / name)
    shutil.copy2(Path(__file__).resolve(), out_dir / Path(__file__).name)

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    print("packaging zip ...")
    if zip_path.exists():
        zip_path.unlink()
    make_zip(out_dir, zip_path)

    print()
    print("summary")
    print(f"  family        : {FAMILY_DEFAULT} {STYLE_DEFAULT}")
    print(f"  glyphs        : {len(glyphs)}  "
          f"(U+{glyphs[0].cp:04X}-U+{glyphs[-1].cp:04X})")
    print(f"  ascii         : {len(ascii_glyphs)}  "
          f"({half_count} half-width 8 px, {full_count} full-width 16 px "
          f"incl. U+3000)")
    print(f"  pua           : {len(pua_glyphs)}")
    print(f"  skipped       : {len(skipped)}")
    print(f"  ttf           : {ttf_path}  ({ttf_path.stat().st_size:,} bytes)")
    print(f"  hex           : {hex_path}  ({hex_path.stat().st_size:,} bytes)")
    if woff2_path.exists():
        print(f"  woff2         : {woff2_path}  ({woff2_path.stat().st_size:,} bytes)")
    if samples_ok:
        print(f"  samples       : {preview_path.name}, {preview_html_path.name}")
    else:
        print("  samples       : FAILED (font and zip were still written)")
    if table is not None:
        missing_count = len(set(table) - {g.cp for g in pua_glyphs})
        print(f"  coverage      : {len(pua_glyphs)}/{len(table)} table code points "
              f"({missing_count} missing)")
    print(f"  sha256(ttf)   : {sha256(ttf_path)}")
    print(f"  zip           : {zip_path}  ({zip_path.stat().st_size:,} bytes)")
    return 0 if samples_ok else 3


if __name__ == "__main__":
    sys.exit(main())
