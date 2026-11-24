#!/usr/bin/env python3
"""Multi-engine benchmark, scored against ground truth where it exists.

Every engine gets the same image, its SVG is rendered by the same engine
(cairosvg) and its editable points are counted by the same metric. For
fixtures rasterised from an SVG original, output size is reported as a
multiple of what the designer actually drew.

    python3 compare.py                              # all engines, all images
    python3 compare.py --engines v1,v2 --gt-only    # ground-truth fixtures only
    python3 compare.py --only 12,14

Writes compare-out/COMPARISON.md with a summary table and per-image detail.
"""

import argparse
import json
import math
import os
import time

from vectorlab.core.metrics import count_editable_points
from verify import score

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _engine_v1(image_path, svg_path):
    from vectorlab import VectorizeOptions, vectorize_image

    result = vectorize_image(image_path, VectorizeOptions(color_count=6))
    with open(svg_path, "w", encoding="utf-8") as handle:
        handle.write(result.svg)


def _engine_v2(image_path, svg_path):
    from vectorlab.trace import vectorize_v2

    result = vectorize_v2(image_path)
    with open(svg_path, "w", encoding="utf-8") as handle:
        handle.write(result.svg)


def _engine_vtracer(image_path, svg_path):
    import vtracer

    vtracer.Config().convert_file(image_path, svg_path)


def _engine_vtracer_simplify(image_path, svg_path):
    import vtracer

    vtracer.Config(simplify=2.0).convert_file(image_path, svg_path)


def _v2_variant(**changes):
    def engine(image_path, svg_path):
        from vectorlab.trace import V2Options, vectorize_v2

        result = vectorize_v2(image_path, V2Options(**changes))
        with open(svg_path, "w", encoding="utf-8") as handle:
            handle.write(result.svg)

    return engine


ENGINES = {
    "v1": _engine_v1,
    "v2": _engine_v2,
    "vtracer": _engine_vtracer,
    "vtracer-simplify": _engine_vtracer_simplify,
    # ablation: v2 with one idea switched off
    "v2-no-coverage": _v2_variant(use_coverage=False),
    "v2-greedy-fit": _v2_variant(use_optimal_fit=False),
    "v2-no-primitives": _v2_variant(use_primitives=False),
    "v2-no-stacking": _v2_variant(use_stacking=False),
    "v2-no-occlusion": _v2_variant(use_occlusion=False),
    "v2-no-blend-labels": _v2_variant(use_blend_labels=False),
    "v2-no-gradients": _v2_variant(use_gradients=False),
    "v2-no-strokes": _v2_variant(use_strokes=False),
    # tolerance sweep
    "v2-tol-0.4": _v2_variant(tolerance=0.4),
    "v2-tol-0.8": _v2_variant(tolerance=0.8),
    "v2-tol-1.0": _v2_variant(tolerance=1.0),
    "v2-tol-1.3": _v2_variant(tolerance=1.3),
    "vtracer-simplify-1": lambda image, svg: __import__("vtracer").Config(simplify=1.0).convert_file(image, svg),
    "vtracer-simplify-3": lambda image, svg: __import__("vtracer").Config(simplify=3.0).convert_file(image, svg),
}


def _load_ground_truth(folder):
    path = os.path.join(folder, "ground-truth.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _measure(engine_name, image_path, svg_path):
    started = time.perf_counter()
    try:
        ENGINES[engine_name](image_path, svg_path)
    except Exception as error:  # noqa: BLE001 - one engine failing shouldn't stop the run
        return {"engine": engine_name, "error": f"{type(error).__name__}: {error}"}
    elapsed = time.perf_counter() - started

    with open(svg_path, encoding="utf-8") as handle:
        counts = count_editable_points(handle.read())
    agreement, delta, _ = score(image_path, svg_path)

    return {
        "engine": engine_name,
        "anchors": counts["anchors"],
        "shapes": counts["shapes"],
        "bytes": os.path.getsize(svg_path),
        "seconds": elapsed,
        "agreement": agreement * 100.0,
        "delta": delta,
        "error": "",
    }


def _geometric_mean(values):
    values = [value for value in values if value > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _summary_lines(engines, results, ground_truth):
    lines = [
        "## Summary — ground-truth fixtures",
        "",
        "`× original` is the geometric mean of output anchors divided by the "
        "designer's anchors. **1.0× would match what the designer drew.**",
        "",
        "Mean error is the average per-channel colour difference (0–255) between "
        "the source and the rendered SVG — lower is more faithful, and unlike "
        "agreement it doesn't saturate near 100%.",
        "",
        "| Engine | × original | Mean error | Mean agreement | Worst agreement | Mean time |",
        "|---|---|---|---|---|---|",
    ]
    for engine in engines:
        rows = [
            row
            for name, row in results
            if row["engine"] == engine and not row["error"] and name in ground_truth
        ]
        if not rows:
            continue
        ratios = [row["ratio"] for row in rows]
        agreements = [row["agreement"] for row in rows]
        deltas = [row["delta"] for row in rows]
        seconds = [row["seconds"] for row in rows]
        lines.append(
            f"| {engine} | {_geometric_mean(ratios):.2f}× | {sum(deltas) / len(deltas):.2f} "
            f"| {sum(agreements) / len(agreements):.1f}% | {min(agreements):.1f}% "
            f"| {sum(seconds) / len(seconds):.2f}s |"
        )
    lines.append("")
    return lines


def _groups(entry):
    """The buckets one fixture's result counts towards."""
    keys = []
    if entry.get("category"):
        keys.append(("category", entry["category"]))
    keys.append(("input", "jpeg" if entry.get("variant") == "jpeg" else "png"))
    features = entry.get("features")
    if features is not None:
        keys.append(("features", "+".join(features) if features else "plain fills"))
    return keys


def _breakdown_lines(engines, results, ground_truth):
    """× original and error per category, per input type and per feature set."""
    buckets = {}
    for name, row in results:
        entry = ground_truth.get(name)
        if not entry or row["error"]:
            continue
        for key in _groups(entry):
            buckets.setdefault(key, {}).setdefault(row["engine"], []).append(row)
    if len({key for key in buckets if key[0] != "input"}) < 1 and len(buckets) < 2:
        return []

    lines = ["## Breakdown", "", "| Group | Engine | Fixtures | × original | Mean error |", "|---|---|---|---|---|"]
    for (kind, value), per_engine in sorted(buckets.items()):
        for engine in engines:
            rows = per_engine.get(engine)
            if not rows:
                continue
            ratio = _geometric_mean([row["ratio"] for row in rows])
            error = sum(row["delta"] for row in rows) / len(rows)
            lines.append(f"| {kind}: {value} | {engine} | {len(rows)} | {ratio:.2f}× | {error:.2f} |")
    lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-engine vectorizer benchmark")
    parser.add_argument("--input", default="samples/bench")
    parser.add_argument("--out", default="compare-out")
    parser.add_argument("--engines", default="v1,vtracer,vtracer-simplify")
    parser.add_argument("--only", default="", help="comma-separated name filters")
    parser.add_argument("--gt-only", action="store_true", help="only fixtures with ground truth")
    parser.add_argument("--split", choices=["tune", "test"], help="only fixtures in this split")
    args = parser.parse_args(argv)

    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in engines if name not in ENGINES]
    if unknown:
        parser.error(f"unknown engines: {', '.join(unknown)} (have {', '.join(ENGINES)})")

    ground_truth = _load_ground_truth(args.input)
    names = sorted(
        name
        for name in os.listdir(args.input)
        if name.lower().endswith(IMAGE_SUFFIXES) and not name.startswith(".")
    )
    if args.only:
        tokens = [token.strip() for token in args.only.split(",") if token.strip()]
        names = [name for name in names if any(token in name for token in tokens)]
    if args.gt_only:
        names = [name for name in names if name in ground_truth]
    if args.split:
        names = [name for name in names if ground_truth.get(name, {}).get("split") == args.split]

    os.makedirs(args.out, exist_ok=True)
    results = []
    detail = ["## Per image", ""]

    for name in names:
        image_path = os.path.join(args.input, name)
        stem = os.path.splitext(name)[0]
        truth = ground_truth.get(name)
        heading = f"### {name}"
        if truth:
            heading += f" — original: {truth['shapes']} shapes, {truth['anchors']} anchors"
        print(f"\n{heading[4:]}")
        detail += [heading, "", "| Engine | Shapes | Anchors | × original | Agreement | Time |",
                   "|---|---|---|---|---|---|"]

        for engine in engines:
            svg_path = os.path.join(args.out, f"{stem}.{engine}.svg")
            row = _measure(engine, image_path, svg_path)
            if row["error"]:
                print(f"  {engine:17s} error — {row['error']}")
                detail.append(f"| {engine} | — | — | — | error | — |")
                results.append((name, row))
                continue

            row["ratio"] = row["anchors"] / float(truth["anchors"]) if truth else 0.0
            ratio_text = f"{row['ratio']:.1f}×" if truth else "—"
            print(
                f"  {engine:17s} {row['shapes']:5d} shapes {row['anchors']:7d} anchors "
                f"{ratio_text:>7s}  {row['agreement']:5.1f}%  {row['seconds']:6.2f}s"
            )
            detail.append(
                f"| {engine} | {row['shapes']} | {row['anchors']} | {ratio_text} "
                f"| {row['agreement']:.1f}% | {row['seconds']:.2f}s |"
            )
            results.append((name, row))
        detail.append("")

    summary = _summary_lines(engines, results, ground_truth) + _breakdown_lines(engines, results, ground_truth)
    report = os.path.join(args.out, "COMPARISON.md")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(["# Vectorizer comparison", ""] + summary + detail) + "\n")

    print()
    print("\n".join(summary))
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
