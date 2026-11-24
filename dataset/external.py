#!/usr/bin/env python3
"""Real designer SVGs -> raster fixtures with ground truth and a held-out split.

Sources, read only, never executed:

    dataset/external/**/*.svg                         anything you save there;
                                                      a subfolder name becomes its category
    dataset/vector_image_to_vector_benchmark/assets/  the Wikimedia starter set,
                                                      with licences from its manifest

Each SVG is rendered at 1024 px on its longest edge, plus a JPEG copy,
into samples/external/. Ground truth records anchors, shapes, the
features the file uses (strokes, gradients, reuse, text, clipping), its
licence where known, and its split:

    tune  ~60%  what the engine may be adjusted against
    test  ~40%  held out; only for reporting

The split comes from a hash of the filename, so it is fixed: nobody can
quietly move a hard file out of the test set.

    python3 dataset/external.py
    python3 compare.py --input samples/external --split test --engines v2,vtracer-simplify
"""

import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from vectorlab.core.metrics import analyse_svg  # noqa: E402

LONGEST_EDGE = 1024
MAX_BYTES = 8 * 1024 * 1024
OUT = os.path.join(ROOT, "samples", "external")
STARTER = os.path.join(HERE, "vector_image_to_vector_benchmark")


def split_for(name):
    digest = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
    return "tune" if digest % 10 < 6 else "test"


def _safe(text):
    """Refuse documents that need anything beyond their own content to render."""
    head = text[:16384]
    if "<!ENTITY" in head or "<!DOCTYPE" in head and "[" in head.split("<!DOCTYPE", 1)[1][:200]:
        return "declares XML entities"
    if re.search(r'(?:href|src)\s*=\s*["\'](?:https?:|file:|//)', text):
        return "references external resources"
    if "<script" in text:
        return "contains script"
    return None


def _rasterise(text, png_path):
    import cairosvg
    from PIL import Image

    first = cairosvg.svg2png(bytestring=text.encode("utf-8"), unsafe=False, background_color="white")
    width, height = Image.open(io.BytesIO(first)).size
    scale = LONGEST_EDGE / float(max(width, height))
    cairosvg.svg2png(
        bytestring=text.encode("utf-8"), unsafe=False, background_color="white",
        scale=scale, write_to=png_path,
    )
    image = Image.open(png_path).convert("RGB")
    return image


def _sources():
    """(path, category, licence record) for every candidate SVG."""
    found = []
    external = os.path.join(HERE, "external")
    if os.path.isdir(external):
        for folder, _, files in os.walk(external):
            category = os.path.relpath(folder, external)
            category = "uncategorised" if category == "." else category.split(os.sep)[0]
            for name in sorted(files):
                if name.lower().endswith(".svg"):
                    found.append((os.path.join(folder, name), category, {}))

    manifest_path = os.path.join(STARTER, "manifest.json")
    assets = os.path.join(STARTER, "assets")
    if os.path.exists(manifest_path) and os.path.isdir(assets):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for item in manifest.get("assets", []):
            path = os.path.join(assets, item.get("filename", ""))
            if os.path.isfile(path):
                record = {key: item.get(key) for key in ("author", "license", "source_page")}
                found.append((path, "wikimedia-starter", record))
    return found


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Real SVGs -> fixtures with ground truth")
    parser.add_argument("--from", dest="source", help="ingest this folder instead of the defaults")
    parser.add_argument("--out", default=OUT, help="where fixtures are written")
    args = parser.parse_args(argv)
    out = args.out

    if args.source:
        sources = [
            (os.path.join(folder, name), os.path.basename(folder) or "uncategorised", {})
            for folder, _, files in os.walk(args.source)
            for name in sorted(files)
            if name.lower().endswith(".svg")
        ]
    else:
        sources = _sources()
    if not sources:
        print("No SVGs found. Save files under dataset/external/, or run")
        print("dataset/vector_image_to_vector_benchmark/download_assets.py on your own machine.")
        return 1

    os.makedirs(out, exist_ok=True)
    truth, skipped = {}, []
    for path, category, record in sources:
        name = os.path.basename(path)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.path.splitext(name)[0]).strip("-")[:80]
        if os.path.getsize(path) > MAX_BYTES:
            skipped.append((name, "larger than 8 MB"))
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        problem = _safe(text)
        analysis = analyse_svg(text) if problem is None else None
        if problem is None and analysis is None:
            problem = "could not be parsed"
        if problem:
            skipped.append((name, problem))
            continue
        if analysis["shapes"] == 0:
            skipped.append((name, "no shapes the metric can count"))
            continue

        try:
            image = _rasterise(text, os.path.join(out, f"{stem}.png"))
        except Exception as error:  # noqa: BLE001 - one bad file shouldn't stop the batch
            skipped.append((name, f"render failed: {error}"))
            continue
        image.save(os.path.join(out, f"{stem}-jpeg.jpg"), "JPEG", quality=82)

        entry = {
            "source": os.path.relpath(path, ROOT),
            "category": category,
            "split": split_for(stem),
            "width": image.width,
            "height": image.height,
            "anchors": analysis["anchors"],
            "shapes": analysis["shapes"],
            "features": analysis["features"],
            **{key: value for key, value in record.items() if value},
        }
        truth[f"{stem}.png"] = entry
        truth[f"{stem}-jpeg.jpg"] = {**entry, "variant": "jpeg"}
        print(f"{entry['split']:4s}  {category:18s} {stem[:40]:40s} {analysis['anchors']:6d} anchors  {','.join(analysis['features']) or '-'}")

    with open(os.path.join(out, "ground-truth.json"), "w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=1, sort_keys=True)

    originals = [entry for name, entry in truth.items() if entry.get("variant") != "jpeg"]
    tune = sum(entry["split"] == "tune" for entry in originals)
    print(f"\n{len(originals)} SVGs ingested: {tune} tune, {len(originals) - tune} test (each also as JPEG)")
    for name, reason in skipped:
        print(f"skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
