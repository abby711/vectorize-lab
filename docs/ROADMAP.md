# Roadmap

## Done

- Sub-pixel outlines from antialiasing, a detected palette, blend-aware
  labels, regions merged rather than dropped, shapes layered with holes
  filled and extended under the shapes on top, refined corners, the
  fewest curves within tolerance, and circles, ellipses and rectangles
  as primitives.
- Transparency, with partly transparent edges read as coverage.
- Linear and radial gradient fills, including glows that fade out.
- Lines of even width as stroked centrelines.
- Up to 7× faster than at first, with work spread across cores, and a
  looser texture mode for brushwork, engravings and stained glass.
- A benchmark against the original drawings, a comparison with VTracer,
  and property tests over random scenes.

## Next

1. **The designer's file.** Rebuild the structure of the original, not
   just its look: repeated elements as one shape reused with `<use>`,
   sizes and gaps that are nearly equal made equal, edges that are
   nearly aligned or parallel made exactly so, and the palette as named
   swatches. Each one can be scored against the original drawings.
2. **Small images.** At 160 px the colour error is 1.05, against 0.42 for
   VectorCraft's Flat Logo mode.
3. **Transparent edges.** VectorCraft places them more precisely (error
   0.06 against 0.15 on the transparent logo).
4. **Blurry and upscaled artwork.** Edges 2–3 px wide pick up thin bands
   of in-between colour: the badge enlarged 2× traces to 282 anchors
   instead of 46.
5. **Partly transparent fills.** A soft shadow should come out as a fill
   with opacity, not opaque or missing.
6. **A compiled core.** VTracer and VectorCraft are 10–100× faster on
   detailed images. Region labelling, outline tracing and curve fitting
   in Rust or C++ is the way to close that.
7. **More ground truth.** Real designer SVGs (see
   [BENCHMARKS.md](BENCHMARKS.md#your-own-svgs)) would show whether the
   results hold beyond the fixtures drawn for this repo.

## Open questions

- Is 0.6 px the right default tolerance? The tolerance sweep suggests
  anything from 0.6 to 1.0 is reasonable.
- Under the anchor count a circle as `<circle>` costs a point more than
  three curves. It's kept because it's easier to edit, which the metric
  doesn't capture.
