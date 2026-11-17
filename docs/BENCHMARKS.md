# Benchmarks

All numbers here come from `compare.py`. In its output, `v2` is the
current engine, `v1` is the first version, and `vtracer` / `vtracer-simplify`
are VTracer 1.0 at default settings and with `--simplify 2`.

## Method

The benchmark uses SVG files where the original drawing is known. Each
one is rendered to PNG, the PNG is traced, and the traced SVG is compared
with the original in two ways:

- Anchor points: the number of on-curve points in the output, divided by
  the number in the original. 1.0× means the same count as the original
  drawing. Circles, ellipses and rects count as 4 points; shapes reused
  with `<use>` are counted each time they're drawn. See `vectorlab/metrics.py`.
- Colour error: the output is rendered with cairosvg and compared pixel by
  pixel with the input. This is the mean absolute difference per channel,
  on a 0–255 scale.

The tables also show "agreement", the share of pixels within 32 levels
of the input. It saturates near 100%, so colour error is the better
measure for comparing engines.

An earlier version of the metric counted SVG command letters rather than
anchor points, which made VTracer look about three times worse than it
is. Everything below uses the corrected count.

## Designer fixtures

Seven illustrations drawn for this repo: a flat landscape, a brand mark,
a UI card, a low-poly mosaic, organic blobs, a potted plant and weather
icons.

![The seven fixtures](images/ground-truth-fixtures.png)

| Engine | Anchors vs original | Colour error | Worst agreement |
|---|---|---|---|
| v2 | 1.21× | 0.10 | 99.9% |
| VTracer `--simplify 2` | 1.26× | 1.49 | 98.5% |
| VTracer default | 4.10× | 1.15 | 99.1% |
| v1 | 3.77× | 2.42 | 98.4% |

Both tools have a setting that trades points for accuracy, so here they
are across a range of settings. Lower-left is better.

![Anchor points against colour error](images/pareto.png)

At the same point count (1.13×, v2 at tolerance 1.0 and VTracer at
`--simplify 3`), the colour error is 0.16 for v2 and 1.71 for VTracer.

Source pixels, v2 and VTracer, zoomed in:

![Zoomed comparison](images/v2-zoom.png)

![Side by side on four fixtures](images/v2-vs-vtracer.png)

## Complex scenes

Six generated scenes, each aimed at a particular weakness: a city at
night with 254 shapes and stars 1.5–2.5 px across, 60 overlapping
confetti shapes, a grid of 24 icons, stroked line art, and two of the
fixtures above rendered at 160 px and 1440 px. They come from
`dataset/generate.py`, which is seeded.

![Complex scenes](images/complex-v2-vs-vtracer.png)

| Engine | Anchors vs original | Colour error | Worst agreement |
|---|---|---|---|
| v2 | 1.22× | 0.65 | 99.4% |
| VTracer `--simplify 2` | 1.39× | 2.20 | 97.1% |

![Details](images/complex-detail.png)

VTracer rounds the plus signs in the icon grid into four-pointed stars,
and fills the leaves in the line art. v2's plus signs have slightly
bulging ends: the bars are 7.7 px wide with a 2 px corner radius, and the
fitted curve stays within the 0.6 px tolerance but is visible when zoomed
in. `--tolerance 0.3` fixes it at the cost of more points.

A seventh scene, a sunset with gradients, isn't in that table. v2 turns
the gradient into 36 bands of flat colour (99.6% agreement, 16× the
points); VTracer flattens the sky to a single colour (72.7%). Neither is
a good result. It needs proper gradient fills.

## Real images

Nine images downloaded from the web, each scaled to 1024 px and given to
both tools. There's no original drawing here, so only point count and
colour error are compared. Apart from the mushroom, these images aren't
in the repo because their licences are unknown.

| Image | v2 points | v2 error | VTracer points | VTracer error |
|---|---|---|---|---|
| Clip-art mushroom | 1,044 | 3.52 | 2,944 | 7.61 |
| Vector flower | 2,833 | 2.08 | 5,860 | 3.71 |
| Cartoon mascot | 6,320 | 4.22 | 11,191 | 4.98 |
| Isometric city infographic | 13,115 | 3.78 | 22,921 | 6.42 |
| Low-poly fox | 19,578 | 8.24 | 23,730 | 12.15 |
| Neon gradient blobs | 11,795 | 13.28 | 21,183 | 16.12 |
| Graffiti alphabet | 47,209 | 11.76 | 140,143 | 10.76 |
| Botanical engraving | 22,155 | 21.18 | 192,430 | 18.58 |
| Stained-glass peacock | 42,721 | 16.46 | 144,932 | 13.44 |

v2 is ahead on both counts for the first six. On the last three, which
are mostly texture, VTracer is more accurate. Part of the gap comes from
v2's 6,000-region limit: with the limit raised, the graffiti comes out at
53,320 points and an error of 13.4.

## What each part contributes

Each row turns one part of v2 off and reruns the designer fixtures.

| Turned off | Anchors vs original | Colour error |
|---|---|---|
| nothing | 1.21× | 0.12 |
| reading antialiasing (sub-pixel edges) | 2.34× | 0.12 |
| layering shapes, holes filled | 1.92× | 0.19 |
| blend-aware colour labels | 1.45× | 0.12 |
| extending shapes under the ones on top | 1.29× | 0.14 |
| optimal curve fitting (greedy instead) | 1.23× | 0.14 |
| circle/ellipse/rect primitives | 1.18× | 0.14 |

Sub-pixel edges and layering account for most of the improvement. The
optimal fitting only saves about 2% over a greedy fit. Primitives slightly
increase the count, since a circle counts as 4 points and the fitter can
draw one with 3 curves; they're kept because a circle that stays a circle
is easier to edit.

## Reproducing

```bash
pip install cairosvg "vtracer==1.0.0a4"

python3 dataset/generate.py           # regenerate the complex scenes
python3 dataset.py                    # render every SVG in dataset/svg/ and record ground truth
python3 compare.py --gt-only          # v1, v2 and VTracer on every fixture with ground truth
```

Other engines can be passed with `--engines`, for example
`v2,v2-no-coverage,v2-greedy-fit` for the ablation, or
`v2-tol-0.4,v2,v2-tol-1.0,vtracer-simplify-1,vtracer-simplify-3` for the
tolerance sweep. Results go to `compare-out/COMPARISON.md`.

### Your own SVGs

Put SVG files in `dataset/external/` (a subfolder name becomes the
category), then:

```bash
python3 dataset/external.py
python3 compare.py --input samples/external --split tune --engines v2,vtracer-simplify
python3 compare.py --input samples/external --split test --engines v2,vtracer-simplify
```

Each SVG is rendered at 1024 px, plus a JPEG copy. The ground truth notes
which features a file uses (strokes, gradients, text, clipping), and the
results are broken down by them. Files that load external resources,
declare XML entities or contain scripts are skipped.

Files are split into a tune set (about 60%) and a test set (about 40%)
by a hash of the filename, so the split never changes. The idea is to
adjust the engine against `tune` and only report `test`. Nothing in
`dataset/external/` is committed.

`dataset/vector_image_to_vector_benchmark/` has a downloader for seven
SVGs from Wikimedia Commons. Run it yourself and `dataset/external.py`
picks them up. They're CC BY-SA 4.0; attribution is in that folder.
