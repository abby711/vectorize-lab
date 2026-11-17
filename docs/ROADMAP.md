# Roadmap

## Done

- **v2 engine** — sub-pixel contours from antialiasing coverage,
  detected palette, blend-aware labels, merge-never-drop partition,
  stacking with occlusion, refined corners, optimal spline fitting,
  primitive recognition. See [APPROACH.md](APPROACH.md).
- **Ground-truth benchmark** — SVG originals rasterised and traced back,
  scored on anchors against the original and on colour error.
- **Ablation and tolerance sweep** — every idea measured on its own.
- **Property tests** — invariants over seeded random scenes.

## Next, in priority order

1. **More ground truth.** Seven fixtures, all mine. Real designer SVGs
   from CC0 sets (Open Peeps, Open Clipart, unDraw) would show whether
   the results generalise. This matters more than any engine change.
2. **Speed.** The hot paths — marching squares chaining, the span table,
   coverage — are vectorizable further, or portable. A native port is
   where the 50× gap to VTracer closes.
3. **Transparency.** Treat alpha as coverage directly at opaque/clear
   edges and keep holes that contain transparency.
4. **Conflation seams.** Where two shapes abut in the source, the
   renderer leaks background into the seam. Unmixing against three
   colours instead of two would read these edges cleanly.
5. **Strokes as strokes.** Thin uniform-width shapes traced as a
   centreline with a stroke width, the way they were drawn.
6. **Gradients as gradients.** Detect linear and radial ramps and emit
   gradient fills instead of bands.

## Questions still open

- Is 0.6 px the right default tolerance? The sweep says 0.6–1.0 is
  the sweet spot; that should be confirmed on real designer files.
- Primitives cost a few percent of anchors under the current metric. Is
  a circle as `<circle>` worth that? For editing, almost certainly — but
  the metric doesn't capture it.
