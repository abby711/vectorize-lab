# How it works

The aim is an SVG that looks like the input, uses as few anchor points as
possible, and is put together the way a designer would draw it. The
numbers quoted here come from [BENCHMARKS.md](BENCHMARKS.md), which
traces SVGs whose original drawing is known and compares the result with
that drawing.

The engine is `vectorlab/v2/`. `vectorlab/pipeline.py` is the first
version, kept for comparison.

## Compared with VTracer

| Stage | VTracer 1.0 | vectorize-lab |
|---|---|---|
| Colours | clustered, with a colour precision setting | found in the image: histogram peaks that have a solid interior |
| Edge pixels | nearest colour | read as a blend of the two colours either side |
| Small regions | speckle filter | merged into the neighbour whose colour is closest |
| Layering | stacked, no holes | largest first, holes filled, shapes extended under the ones on top |
| Outline | pixel staircase, then smoothed | placed between pixels using the antialiasing |
| Where curves break | inflections and accumulated angle | the fewest curves that stay within the tolerance |
| Shapes | paths | circles, ellipses and rectangles where they fit |
| Gradients | flat colour | linear and radial gradient fills |
| Lines | filled outlines | stroked centrelines |
| Transparency | kept | kept, with partly transparent edges read as coverage |

VTracer traces each colour region's outline along the pixel grid, then
works to smooth the resulting staircase back into curves. This engine
skips the staircase: it works out where each edge actually was from the
antialiased pixels, then fits curves to that.

## Stages

### 1. Load

Images larger than 1024 px are scaled down with a box filter. Other
filters ring around edges, and the later stages assume an edge pixel is
an average of the colours either side.

An image with transparency is composited over a key colour, chosen to
be as far as possible from every colour in the artwork. A partly
transparent edge pixel then becomes a blend of the shape's colour and
the key, which the next stages already know how to read. The key
colour's regions are left out of the SVG.

### 2. Palette

Flat artwork has a few exact colours, each covering a lot of pixels,
while antialiasing blends are spread thinly across many values. The
palette comes from peaks in the colour histogram, and a peak only counts
if it has an interior: some pixels whose four neighbours are the same
colour. Blends live in bands one or two pixels wide and never do.

How close two colours can be and still count as different depends on
the image. Noise is measured as the average difference between
neighbouring pixels inside flat areas (close to 0 for a clean PNG,
around 0.5 for a JPEG) and the tolerances scale with it.

Images without flat colours, like gradients and photographs, get a
k-means palette with as many colours as it takes to bring the average
error under 4 ΔE, up to 96.

### 3. Labels

Each pixel gets a palette colour. A pixel that isn't close to any of
them is checked as a blend of two colours from its neighbourhood and
given whichever it's mostly made of. Without this, the pixels between
blue and yellow would be called green and grow green slivers.

For JPEGs, blends are allowed to reach 3 px from an edge, because JPEG
stores colour at half resolution and smears it.

### 4. Regions

Pixels of the same colour that touch form a region. Regions below a
minimum size (4 px in clean artwork, up to 16 px in a noisy JPEG) are
merged into the neighbour whose colour is closest, so no pixel is ever
left unpainted. A small region that stands out from all its neighbours
is kept, unless it's a blend of two of them.

There's a budget of 6,000 regions. A detailed image over it is
simplified by merging larger regions until it fits, and the result
says so.

### 5. Gradients

A gradient has no flat colours, so the palette cuts it into bands. Where
two bands of one gradient meet, the colour changes across the boundary
by about as much as it does between neighbouring pixels inside each
band. At a real edge, even a blurred one, it changes far more. Bands
joined like that are grouped, and the group becomes one shape if a
linear or radial gradient, with a few colour stops, reproduces its
pixels. If one doesn't, a gradient with radial glows over it that fade
to transparent is tried, then each half of the group on its own.

### 6. Paint order

Shapes are painted largest first, counting the area of their holes.
Anything inside a shape's hole is smaller, so it's painted on top: a
ring becomes a disc with a smaller disc over it, and every shape is a
single outline. A hole that contains any transparency stays a hole.

Each shape is then extended under the shapes painted after it, within
its own convex hull. A designer draws the sky as one rectangle and the
hills over it; tracing only the visible sky would give an outline that
follows every hilltop. The extra area is covered anyway, so the picture
doesn't change. Its outline is hidden, so there's no antialiasing to
read there, and its coverage comes from the hull instead.

A shape whose pixels would only show the same flat colour that's
already beneath it, such as the inside of an outline, is left out.

### 7. Lines

A thin shape is checked for being one line of even width. Its mask is
thinned to a skeleton, and each skeleton point is moved to halfway
between the two sides of the outline, which also gives the width there.
If most of the widths are close to the median, and a line of that width
drawn along the centreline reproduces the shape, it's written as a
stroke. Where lines meet, their ends are moved to where they'd
intersect, and lines that carry straight on through a junction are
joined into one. Each free end gets a round or a flat cap, whichever
fits better.

Lines are painted after all the filled shapes, as outlines are drawn
over their fills, so the shapes they touch can extend under them.

### 8. Outlines

Each edge pixel is split between the colour inside the shape and the
colour across the edge, which gives its coverage: a pixel that's 30%
red and 70% white is 30% covered. Marching squares on that coverage
field places the outline between pixels. Coverage is kept just either
side of 0.5, so the outline's shape is the same as the pixel mask's;
only its position is refined.

Since one shape's coverage against another is one minus the other's, two
neighbouring shapes trace the same line along the edge they share.

On a rendered circle, this puts the outline within 0.03 px of the true
edge on average, against 0.19 px for the pixel mask.

### 9. Corners

Rendering shaves a sharp corner into a short chamfer, and a corner
detector lands on one end of it. A line is fitted to each side and the
corner is moved to where they intersect. A corner that bulges off its
two sides was rounded by the designer and isn't sharpened.

### 10. Curves

Each outline is split at its corners. Along each piece, candidate break
points come from a Douglas–Peucker simplification. Every stretch between
two candidates is fitted with one cubic Bézier or a line, and a dynamic
program picks the fewest pieces that all stay within the tolerance
(0.6 px by default). Among equal counts it picks the one closest to the
outline. The chosen curves are then checked against every outline point,
and any that stray too far are ruled out and the search reruns.

At a smooth break point the tangent is estimated once and shared by the
curves either side, so the join is smooth. At a corner it's left free.

A smooth closed outline has to start somewhere, and that start is
always an anchor. The leftmost, topmost, rightmost and bottommost points
are each tried, and the cheapest kept.

### 11. Shapes

An outline that fits a circle, an ellipse at any angle, or a rectangle
with or without rounded corners is written as that element.

## Texture

Brushwork, engravings and stained glass are detail everywhere rather
than flat shapes. When the pixels inside regions are noisy and at least
30% of the image is in regions under 200 px, the gradient and line
searches are skipped and curves may stray up to 1 px. That roughly
halves the time on such images, and the result carries a note saying it
was traced approximately.

## Speed

Most per-colour work is done once per distinct colour rather than once
per pixel. Curve fitting runs its batches on threads. With more than one
worker (the command-line default is one per core), line detection,
outline tracing, gradient search and curve fitting are spread across
processes. The output is the same either way.

## What each stage is worth

Turning each stage off on its own and rerunning the designer fixtures:

| Turned off | Anchors vs original |
|---|---|
| nothing | 1.20× |
| reading antialiasing | 1.97× |
| layering, holes filled | 1.89× |
| blend-aware labels | 1.51× |
| extending under shapes on top | 1.27× |
| optimal curve fitting | 1.26× |
| primitives | 1.17× |

Reading the antialiasing and layering the shapes account for most of the
improvement. Primitives cost a few points, because a circle counts as 4
and the fitter can draw one with 3 curves, but a circle that stays a
circle is easier to edit.

## Checks

`tests/test_properties.py` checks properties over seeded random scenes
as well as the fixtures: no pixel is left unpainted, the output is
faithful, deterministic and valid, nothing is dropped, outlines are
within 0.1 px of the truth, primitives come back exact, pure noise is
refused, transparency and holes are kept, gradients and lines come back
as gradients and lines, and no fixture gets worse than 1.8× its
original's anchors.
