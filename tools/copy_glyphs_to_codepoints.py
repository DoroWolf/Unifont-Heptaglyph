from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_MAP_URL = "https://windlandneko.github.io/xingyu/xingyu.json"
IMAGE_EXTS = {".png", ".bmp", ".gif", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
CP_RE = re.compile(r"^(?:U\+|u\+|0[xX])?([0-9A-Fa-f]{4,6})$")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def project_root() -> Path:
    """The folder that contains ``tools/``, ``glyphs/`` and ``xingyu.json``."""
    return Path(__file__).resolve().parent.parent


def codepoint_int(raw: object) -> int:
    """Parse ``F78C0`` / ``U+F78C0`` / ``0xF78C0`` into an int."""
    if isinstance(raw, int):
        value = raw
    else:
        text = str(raw).strip()
        match = CP_RE.match(text)
        if not match:
            raise ValueError(f"not a code point: {raw!r}")
        value = int(match.group(1), 16)
    if not (0 <= value <= 0x10FFFF):
        raise ValueError(f"code point out of range: {raw!r}")
    return value


def format_codepoint(value: int) -> str:
    """Format a code point as an upper-case hex file-name stem."""
    width = max(4, (value.bit_length() + 3) // 4)
    return f"{value:0{width}X}"


def load_mapping_document(map_path: Path) -> dict:
    """Read the cached JSON, downloading it first when it is missing."""
    if not map_path.is_file():
        eprint(f"fetching code-point table: {DEFAULT_MAP_URL}")
        request = urllib.request.Request(
            DEFAULT_MAP_URL,
            headers={"User-Agent": "xingyu-codepoint-copier/1.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_bytes(payload)
        eprint(f"cached table: {map_path}  ({len(payload)} bytes)")
    return json.loads(map_path.read_text(encoding="utf-8"))


def extract_entries(document: object) -> List[dict]:
    """Accept the site's ``{entries: [...]}`` shape and simple name->cp maps."""
    if isinstance(document, dict) and isinstance(document.get("entries"), list):
        return document["entries"]
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        return [{"name": key, "cp": value} for key, value in document.items()]
    raise ValueError("unrecognised code-point table format")


def build_name_map(document: object) -> Dict[str, Tuple[int, str]]:
    """Return ``{name: (code_point, canonical_name)}``.

    Lookup is case-insensitive because the source files come from a Windows
    checkout; the table itself has no case-only duplicate names.
    """
    by_name: Dict[str, Tuple[int, str]] = {}
    duplicates: Dict[str, List[str]] = {}

    for entry in extract_entries(document):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        raw_cp = entry.get("cp")
        if not name or raw_cp is None:
            continue
        name = str(name).strip()
        try:
            value = codepoint_int(raw_cp)
        except ValueError as exc:
            eprint(f"  ! ignoring entry {name!r}: {exc}")
            continue

        key = name.casefold()
        if key in by_name and by_name[key][0] != value:
            duplicates.setdefault(key, [by_name[key][1]]).append(name)
            continue
        by_name[key] = (value, name)

    if duplicates:
        eprint("warning: duplicate names in the table (first one kept):")
        for key, names in duplicates.items():
            eprint(f"  {key}: {', '.join(names)}")
    return by_name


def collect_images(src: Path) -> List[Path]:
    files = [p for p in src.glob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.casefold())


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prune_stale(out: Path, manifest: Path, planned: set[Path]) -> int:
    """Delete files written by an earlier run that this run no longer produces.

    Only paths recorded in the previous manifest and located inside *out* are
    touched, so unrelated files are safe.  This matters after a table update,
    when many code points move and the old file names would otherwise linger.
    """
    if not manifest.is_file():
        return 0

    out_resolved = out.resolve()
    removed = 0
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("output") or "").strip()
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = out / path
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in planned or not resolved.is_relative_to(out_resolved):
                    continue
                if resolved.is_file():
                    resolved.unlink()
                    removed += 1
    except (OSError, csv.Error) as exc:
        eprint(f"warning: could not prune from {manifest}: {exc}")
        return removed

    if removed:
        qdir = out / "_unmapped"
        if qdir.is_dir() and not any(qdir.iterdir()):
            try:
                qdir.rmdir()
            except OSError:
                pass
    return removed


# --------------------------------------------------------------------------- #
# copy
# --------------------------------------------------------------------------- #
def main() -> int:
    root = project_root()
    src = root / "glyphs"
    out = root / "glyphs_cp"
    map_path = root / "data" / "xingyu.json"

    if not src.is_dir():
        eprint(f"error: source folder not found: {src}")
        return 1

    try:
        document = load_mapping_document(map_path)
        table = build_name_map(document)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        eprint(f"error: cannot load code-point table: {exc}")
        return 1

    if not table:
        eprint(f"error: no usable entries in {map_path}")
        return 1

    files = collect_images(src)
    if not files:
        eprint(f"error: no bitmaps found in {src}")
        return 1

    version = document.get("version") if isinstance(document, dict) else None
    version_text = f"v{version}, " if version else ""
    print(f"source : {src}  ({len(files)} bitmaps)")
    print(f"table  : {map_path}  ({version_text}{len(table)} names)")
    print(f"output : {out}")

    mapped: List[Tuple[Path, int, str, Path]] = []
    unmatched: List[Path] = []
    conflicts: List[Tuple[Path, str]] = []
    used_targets: Dict[str, Path] = {}

    for path in files:
        key = path.stem.casefold()
        hit = table.get(key)
        if hit is None:
            unmatched.append(path)
            continue

        value, canonical_name = hit
        target_name = format_codepoint(value) + path.suffix.lower()
        target = out / target_name
        target_key = target_name.casefold()

        if target_key in used_targets and used_targets[target_key] != path:
            conflicts.append((path, canonical_name))
            continue
        used_targets[target_key] = path
        mapped.append((path, value, canonical_name, target))

    manifest = out / "codepoint_manifest.csv"
    qdir = out / "_unmapped"
    planned = {target.resolve() for _, _, _, target in mapped}
    planned.update((qdir / path.name).resolve() for path in unmatched)

    stale_removed = prune_stale(out, manifest, planned)

    copied = 0
    overwritten = 0
    for path, value, canonical_name, target in mapped:
        existed = target.exists()
        try:
            safe_copy(path, target)
        except OSError as exc:
            eprint(f"  ! failed to copy {path.name}: {exc}")
            continue
        copied += 1
        overwritten += int(existed)
        suffix = "  (overwrites existing)" if existed else ""
        print(f"  copy {path.name:<24} -> {target.name}{suffix}")

    quarantined: List[Tuple[Path, Path]] = []
    for path in unmatched:
        target = qdir / path.name
        try:
            safe_copy(path, target)
        except OSError as exc:
            eprint(f"  ! failed to copy unmatched {path.name}: {exc}")
            continue
        quarantined.append((path, target))
        print(f"  copy {path.name:<24} -> _unmapped/{target.name}  [no code point]")

    if conflicts:
        eprint("warning: several sources map to the same code point; later ones skipped:")
        for path, name in conflicts:
            eprint(f"  {path.name} -> {name}")

    # Audit trail: every input file and what happened to it.
    out.mkdir(parents=True, exist_ok=True)
    qmap = {src_path: dst for src_path, dst in quarantined}
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "source_name", "codepoint", "character",
             "output", "status", "canonical_name"]
        )
        for path, value, canonical_name, target in mapped:
            writer.writerow([
                str(path), path.name, f"U+{value:04X}",
                chr(value), str(target), "copied", canonical_name,
            ])
        for path in unmatched:
            target = qmap.get(path)
            writer.writerow([
                str(path), path.name, "", "", str(target) if target else "",
                "unmapped", "",
            ])
        for path, name in conflicts:
            writer.writerow([
                str(path), path.name, "", "", "", f"conflict:{name}", name,
            ])

    print()
    print("summary")
    print(f"  bitmaps found     : {len(files)}")
    print(f"  renamed & copied  : {copied}")
    print(f"  overwritten       : {overwritten}")
    print(f"  stale removed     : {stale_removed}")
    print(f"  no code point     : {len(unmatched)}"
          + (f"  -> {out / '_unmapped'}" if quarantined else ""))
    print(f"  conflicts skipped : {len(conflicts)}")
    print(f"  manifest          : {manifest}")
    if unmatched:
        print("  unmapped names    : " + ", ".join(p.name for p in unmatched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
