# Image-to-Vector benchmark: original SVG ground truth

This package is intended for a local coding agent / QA workflow. It contains a manifest of real, designer-authored SVG source files and a downloader. The original SVG is the ground truth; rasterize each SVG to PNG and submit that PNG to your Image-to-Vector implementation, then compare the returned SVG with the source.

## Get the SVG binaries

From this directory, run:

```bash
python3 download_assets.py
```

The SVG files will be saved under `assets/`. The current execution environment could not connect to Wikimedia Commons, so the package includes the source manifest and downloader rather than pretending the binaries were downloaded.

## Suggested evaluation workflow

1. Keep each original SVG unchanged in `assets/`.
2. Rasterize it at a fixed size (for example, 1024 px on the longest edge) using a consistent renderer.
3. Feed only the PNG to the feature under test.
4. Save the generated SVG beside the PNG, e.g. `outputs/mandala_drawing_01.generated.svg`.
5. Compare side-by-side and at 400% zoom. Also inspect path count, node count, groups, fills, strokes, clipping, gradients, and editability.
6. Do not use pixel similarity alone: compare whether the generated SVG has sensible editable objects and designer-quality paths.

## Included coverage

- Five mandala SVGs: radial/four-fold symmetry, repeated geometry, fine paths, nested regions.
- One abstract botanical pattern: organic curves and irregular shapes.
- One additional mandala illustration: multi-shape segmentation.

This is a starter set, not a balanced benchmark: it is currently weighted toward mandalas. Expand it with authored logos, icons, character illustrations, posters, maps, and multi-object scenes for broader coverage.

See `manifest.json` and `ATTRIBUTION.md` for asset details and source links.
