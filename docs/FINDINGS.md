# Findings log — the v1 era

> **This is the original investigation, kept as a record.** It covers
> how Figma's Vectorize works and the first tracer (v1). Some
> conclusions here were later corrected — notably the anchor-count
> comparison with VTracer, which used a metric that counted SVG command
> letters rather than anchor points. For the current engine and current
> numbers, see [APPROACH.md](APPROACH.md).

---

## Part 1 — Who Figma actually uses

**Answer: Recraft, Inc.**

This is from Figma's own sub-processor disclosure (last updated 30 April
2026), which is a legally-binding list, not marketing copy. Under
**AI Service Providers**:

| Sub-processor | Applicable Service | Processing Activity |
|---|---|---|
| OpenAI, LLC | Figma Design; FigJam; Figma Slides | AI model provider |
| **Recraft, Inc.** | **Figma Design** | **AI model provider** |
| Anthropic PBC | Figma Make | AI model provider |
| Google LLC (Vertex) | Figma Make | AI model provider |
| Jasper AI, Inc. | Figma Make | AI model provider |
| Cerebras / Fireworks / Groq | Figma Make | AI infrastructure |

Only **two** vendors are scoped to *Figma Design*: OpenAI and Recraft.
OpenAI ships no raster-to-SVG product. Recraft's flagship capability is
exactly that — `recraft-vectorize`, "convert raster images to
high-quality SVG format… clean vector paths", ~1M runs on Replicate
alone, and Recraft's own marketing says output is "fully compatible with
Adobe Illustrator, Figma, and Sketch." Recraft was a Figma plugin
partner *before* Figma shipped Vectorize natively in Feb 2026.

So the chain is:

```
Figma client  --(image bytes, AI credits metered)-->  Figma backend
              <--(SVG)---------------------------------  Recraft vectorize API
```

Which matches the SVG-over-the-wire response seen in the network tab.

### Caveats on that conclusion

- The sub-processor list proves Recraft processes Figma Design content.
  It does **not** name which specific feature. Recraft could in
  principle be behind a different Figma Design AI feature and Vectorize
  could be in-house. But Recraft has no other product that maps onto a
  Figma Design feature, so this is a strong inference, not a guess.
- Figma has *not* published a "powered by Recraft" statement. Treat this
  as well-evidenced attribution, not an official confirmation.

### The one open question worth answering yourself

In DevTools, drag the **Recolor → Number of colors** slider and watch
the network panel:

- **New request fires per change** → the vendor re-traces server-side,
  and colour count is a request parameter.
- **No request** → Figma got back a richer full-fidelity result once and
  is re-quantising the returned paths locally.

That single observation tells you where to put the client/server
boundary in our own version. My bet is the second: the help docs say
Vectorize costs credits once, then Recolor is a live panel you commit
with a checkmark — that reads like local re-parameterisation.

### Why the docs give away that the core is geometric, not generative

Figma's help page exposes exactly three recolor modes: *Full color*
(→ "Number of colors"), *Grayscale* (→ "Number of grays"), *Black and
white* (→ "Threshold"). Those are `k` in a colour quantiser and a
binarisation cutoff — classical tracer knobs. The same page also warns
that low-resolution images vectorise badly and suggests running Boost
resolution first, which only matters for a pixel-grid contour tracer.
Recraft brands it as AI (and there is likely a learned pre-pass for
cleanup), but the geometry stage is recognisably Potrace/VTracer-shaped.

**Sources**

- <https://www.figma.com/sub-processors/> — the decisive one
- <https://help.figma.com/hc/en-us/articles/38031452710807-Convert-static-images-to-vector-layers>
- <https://www.figma.com/blog/introducing-vectorize/>
- <https://replicate.com/recraft-ai/recraft-vectorize>
- <https://www.recraft.ai/recraft-integrations/figma>
- <https://github.com/visioncortex/vtracer>, <https://vectorizer.ai/api>

---

## Part 2 — The prototype

Pure `numpy` + `Pillow`, no scipy/skimage/sklearn. Every stage is
deliberately written to port cleanly to C++17.

```
decode → quantize → label regions → extract rings
       → simplify → fit curves → emit SVG
```

| Stage | File | Algorithm |
|---|---|---|
| Palette reduction | `vectorlab/quantize.py` | k-means++ in CIE Lab; grayscale banding; luma threshold |
| Region segmentation | `vectorlab/regions.py` | run-based 4-connected CCL with union-find |
| Contour extraction | `vectorlab/regions.py` | pixel-grid unit-edge walk → exact integer staircase rings |
| Simplification | `vectorlab/simplify.py` | Douglas-Peucker, seam-anchored on an extreme vertex |
| Corners + curves | `vectorlab/curves.py` | turn-angle corner detection, Catmull-Rom → cubic Bezier |
| Emission | `vectorlab/svg.py` | one `<path>` per region, `fill-rule="evenodd"` so holes work free |

### Usage

```bash
python3 make_test_image.py samples          # generate test rasters

# black & white — the hand-lettering / pencil-sketch case
python3 vectorize.py samples/test_sketch.png -o out.svg \
    --mode monochrome --threshold 150

# full colour
python3 vectorize.py samples/test_shapes.png -o out.svg \
    --mode full_color --colors 8

# grayscale banding
python3 vectorize.py in.png -o out.svg --mode grayscale --grays 5

# see the trace stage without curve fitting
python3 vectorize.py in.png -o out.svg --no-fit
```

### As a service (mirrors Figma's architecture)

```bash
python3 server.py                           # 127.0.0.1:8787
curl -X POST --data-binary @in.png \
     "http://127.0.0.1:8787/vectorize?mode=full_color&colors=5" -o out.svg
```

Responds `image/svg+xml` with `X-Shape-Count` / `X-Segment-Count`
headers. Stdlib-only and single-threaded — it exists to pin down the
client/server boundary, not to serve traffic.

### Verification

`verify.py` flattens our own Bezier output, fills it with PIL, and
scores it per-pixel against the source. No SVG renderer needed because
we emit a closed `M/L/C/Z` subset.

```bash
python3 verify.py samples/test_shapes.png out.svg render.png
```

Measured on the generated test rasters:

| Case | Pixels within tolerance | Mean channel delta | Segments |
|---|---|---|---|
| Flat shapes, full colour (k=5) | 99.4% | 1.6 | 98 |
| Flat shapes, polygons only | 99.5% | 1.4 | 226 |
| Soft sketch, grayscale (4 bands) | 97.4% | 3.2 | 858 |
| Soft sketch, monochrome | 93.1% | 5.6 | 336 |

Holes render correctly (the even-odd ring stacking is exercised by the
donut in `test_shapes.png`).

### Real-world run: clip-art mushroom (350x350 JPG)

```bash
IN=samples/cute-clip-art-style-mushroom-illustration-with-white-background-vector.jpg
python3 vectorize.py $IN -o samples/mush_up2.svg \
    --mode full_color --colors 6 --upscale 2.0 --max-dim 1200 --min-area 20
```

| Config | Shapes | Segments | SVG size | Pixel score |
|---|---|---|---|---|
| k=6, no upscale | 68 | 1221 | 25 KB | 94.0% |
| k=8, no upscale | 72 | 1101 | 24 KB | 93.3% |
| k=12, no upscale | 101 | 1253 | 27 KB | 92.6% |
| **k=6, upscale 2x** | **110** | **2053** | **56 KB** | **91.6%** |
| monochrome, thr 110 | 44 | 1551 | 40 KB | 74.3% |

**The winner by eye is the one with the worst score.** At native 350px
the black outline is only 2-3px wide, so simplification eats it and the
outline visibly fragments. Upscaling 2x before tracing thickens it in
pixel terms and the outline comes back continuous — exactly why Figma's
docs tell you to run Boost resolution first. But the pixel-agreement
score *drops*, because the metric is dominated by antialiased edge
pixels that no quantiser can ever match.

Takeaway: `verify.py` is a regression guard against geometry bugs, not a
quality metric. Do not tune against it. Two further observations from
this image:

- Raising `k` past ~6 spends colours on JPEG ringing around the outline,
  not on real detail — shape count climbs, quality doesn't.
- Monochrome luma-thresholds the red cap to black, since saturated red
  has low luminance. Fine for pencil sketches, wrong for coloured line
  art. A production version would threshold on lightness after a
  saturation-aware transform.

### Real-world run: photographic portrait (`samples/RK.jpg`, 400x400)

The documented failure case, reproduced. Figma's own help page warns
"complex or low-resolution images may not vectorize well", and a
continuous-tone photo is the worst input for any region-based tracer.

| Config | Shapes | Segments | Size | Pixel score |
|---|---|---|---|---|
| k=8 | 196 | 3702 | 61 KB | 90.1% |
| k=12 | 306 | 5166 | 82 KB | 90.1% |
| k=16 | 375 | 6031 | 98 KB | 93.2% |
| k=24 | 514 | 7340 | 118 KB | 92.7% |
| poster: k=6, eps 2.8, min-area 200 | 65 | 1519 | 31 KB | 81.7% |
| grayscale, 4 bands | 57 | 2477 | 33 KB | 71.1% |
| **k=16, presmooth 3, min-area 4** | **562** | **6017** | **99 KB** | **95.7%** |

None of these are good, and no parameter choice makes them good. The
reason is structural, not a tuning failure:

- **Smooth gradients have no boundaries to find.** Skin shading is a
  continuous ramp; quantising it invents hard edges that were never in
  the image. Faces land in the uncanny zone — the k=16 pass reads as
  skin blotches, not a face.
- **Bokeh backgrounds explode the region count.** Out-of-focus foliage
  produces hundreds of tiny disconnected blobs, which is where most of
  those 375 shapes and 98 KB go. They carry no useful structure.
- **Raising k does not converge.** Detail per shape stays flat while
  shape count grows linearly. k=24 scores *worse* than k=16.

#### Where the error actually comes from

Measured by isolating each stage against the same 400x400 source at
k=16. The quantised *raster* — before any tracing — is the ceiling any
16-colour tracer can reach:

| Stage | Score | Cost of this stage |
|---|---|---|
| Quantised raster, k=16 (the ceiling) | 98.5% | — |
| Exact polygons, no simplify, all regions | 95.4% | 3.1 pts — **measurement artifact** |
| + `min-area 20` drops small regions | 92.0% | **3.4 pts — the real defect** |
| + simplify at eps 1.6 | 93.1% | −1.1 pts (helps) |
| + curve fitting | 93.3% | 0.2 pts (negligible) |

Three findings, one of which corrects an earlier note in this file:

1. **Colour quantisation is *not* the dominant error.** At k=16 the
   quantised raster already scores 98.5%, and at k=24 it reaches 99.8%.
   An earlier revision of this README claimed quantisation dominated —
   that was wrong. The geometry stages lose more, and lose *more* as k
   rises.
2. **`min-area` was the biggest real loss.** Discarding sub-threshold
   regions leaves unpainted holes — those are the white speckles in the
   hair and beard. Dropping the floor from 20 to 4 and adding a median
   pre-filter takes the same image from **93.3% to 95.7%** at
   *identical* segment count (6031 -> 6017). See `--presmooth`.
3. **About 3 points is `verify.py` itself,** not the SVG. Even
   mathematically exact polygons score 95.4% rather than 98.5%, because
   PIL's polygon fill and our pixel-square ring convention disagree by
   half a pixel on every boundary. Another reason not to tune against
   this metric.

#### Why it looks blocky, which is a separate question

The blockiness is not curve-fitting failing to run — it is curve fitting
having nothing to work with. Diagnostics: 84.5% of retained vertices
exceed the 42-degree corner threshold, so tangents get zeroed and we
emit straight lines. Widening the corner window to 2 only moves that to
74.4%, and pre-smoothing does not move it at all.

The cause is the pipeline order. Douglas-Peucker *keeps precisely the
high-deviation vertices and discards the smooth runs*, so by
construction most survivors look like corners to any local angle test.
Lowering epsilon does not help either — eps 0.5 quadruples segments to
26662 and scores *worse* (91.9%).

The real fix is architectural: Potrace does not simplify-then-connect,
it runs least-squares Bezier fitting against the original staircase ring
under an error bound. That is gap #1 in the list below, and this is the
evidence for why it matters.

#### Conclusion

**Don't vectorize photographs.** A tracer's job
is flat-ish art with real boundaries — logos, lettering, sketches, clip
art. The mushroom clip-art above works because its boundaries exist.

If a photo has to go through, the least-bad path is to stop pretending
to be faithful and commit to a graphic treatment: low k, high epsilon,
high min-area (the "poster" row). It scores worst by pixel agreement and
looks the most deliberate — 65 shapes and 31 KB instead of 375 and
98 KB. Subject isolation (background removal) first would matter more
than any tracer parameter, which is precisely why Figma ships Remove
background next to Vectorize and tells you to run it first.

### Tuning notes

- `--epsilon 1.6` is the default because it halves segment count versus
  `1.0` at identical pixel accuracy. Below ~1.0 you keep the pixel
  staircase; above ~3.0 curves start cutting corners.
- `--corner 42` degrees is the sharp/smooth split. Lower it for
  geometric art, raise it for organic sketches.
- `--smoothness 0` degenerates to polygons; `1.0` is Catmull-Rom
  standard.
- `--max-dim` caps the longest edge before tracing. CCL and the ring
  walk are the cost centres, both roughly linear in pixel count.
- `--upscale 2.0` is the fix whenever the source has strokes thinner
  than ~4px. Costs roughly 2x the segments. Raise `--max-dim` alongside
  it or the downscale cap cancels it out.

### Known gaps versus a production tracer

1. **No Schneider least-squares curve fitting.** Catmull-Rom passes
   through every simplified vertex, so quality is bounded by the
   simplifier. Real fitting would cut node count again.
2. **No geometric primitive snapping.** Circles stay as Bezier
   approximations rather than being recognised as circles. This is a big
   part of why Vectorizer.AI output looks cleaner than VTracer's.
3. **No tangent continuity enforcement** across segment joins.
4. **Adjacent regions share no edges** — abutting colour regions are
   traced independently, so sub-pixel seams can appear. A production
   version traces a shared planar graph once.
5. **No neural pre-pass** (denoise / matte / upscale).

Items 1-3 are the "Tier 2" polish that separates auto-traced from
designed. Item 4 is the one that will bite first on real illustrations.
