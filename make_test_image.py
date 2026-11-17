#!/usr/bin/env python3
"""Generates the benchmark fixture set.

Synthetic on purpose: reproducible, regenerable, and free of any
licence question. Each fixture isolates one axis the tracer is known to
be sensitive to, so a regression shows up as a specific image getting
worse rather than an average drifting.

    python3 make_test_image.py            # -> samples/bench/
    python3 make_test_image.py <dir>
"""

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

SIZE = (480, 360)


def flat_shapes(path):
    """Baseline: flat fills, hard edges, one hole. Should score ~99%."""
    image = Image.new("RGB", SIZE, (245, 242, 236))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 220, 220), fill=(212, 78, 62))
    draw.rectangle((240, 60, 420, 180), fill=(46, 92, 138))
    draw.polygon([(120, 330), (240, 200), (360, 330)], fill=(238, 186, 66))
    draw.ellipse((110, 110, 150, 150), fill=(245, 242, 236))
    image.save(path)


def outlined_icon(path):
    """Clip-art style: flat fills behind a black outline of varying weight.

    The thin-stroke case. Outlines under ~4px get eaten by
    simplification unless the input is upscaled first.
    """
    image = Image.new("RGB", SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.ellipse((60, 80, 260, 260), fill=(226, 92, 76), outline=(20, 18, 16), width=5)
    draw.ellipse((120, 130, 160, 170), fill=(255, 255, 255), outline=(20, 18, 16), width=2)
    draw.ellipse((185, 185, 215, 215), fill=(255, 255, 255), outline=(20, 18, 16), width=2)

    draw.rounded_rectangle(
        (290, 110, 420, 240), radius=18, fill=(104, 154, 92), outline=(20, 18, 16), width=3
    )
    draw.line((300, 290, 420, 290), fill=(20, 18, 16), width=2)
    draw.line((300, 305, 380, 305), fill=(20, 18, 16), width=1)
    image.save(path)


def line_sketch(path):
    """Soft pencil strokes. The monochrome / threshold case."""
    image = Image.new("L", SIZE, 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((90, 70, 380, 290), outline=40, width=9)
    draw.arc((150, 130, 330, 250), start=200, end=340, fill=30, width=7)
    draw.line((240, 40, 240, 90), fill=30, width=6)
    for index in range(6):
        offset = index * 14
        draw.line((120 + offset, 300, 150 + offset, 330), fill=60, width=3)
    image = image.filter(ImageFilter.GaussianBlur(1.6))
    image.convert("RGB").save(path)


def logo_mark(path):
    """Geometric primitives. Tests whether circles survive as circles."""
    image = Image.new("RGB", SIZE, (18, 26, 32))
    draw = ImageDraw.Draw(image)
    centre = (170, 180)
    for radius, colour in ((110, (92, 186, 210)), (76, (18, 26, 32)), (44, (238, 226, 192))):
        draw.ellipse(
            (centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius),
            fill=colour,
        )
    for index in range(5):
        left = 300 + index * 32
        height = 60 + index * 34
        draw.rectangle((left, 300 - height, left + 20, 300), fill=(238, 226, 192))
    image.save(path)


def smooth_gradient(path):
    """Continuous tone with no real boundaries. The known failure case.

    Stands in for a photograph: quantising this invents hard edges that
    were never in the image.
    """
    width, height = SIZE
    image = Image.new("RGB", SIZE)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            distance = math.hypot(x - width * 0.38, y - height * 0.42) / (width * 0.55)
            falloff = max(0.0, 1.0 - distance)
            pixels[x, y] = (
                int(40 + 180 * falloff + 30 * (x / width)),
                int(60 + 120 * falloff**1.4),
                int(90 + 70 * (1.0 - y / height)),
            )
    image.save(path)


def grain_texture(path):
    """Fine noise. The speckle case - how many junk regions appear."""
    import random

    random.seed(11)
    width, height = SIZE
    image = Image.new("L", SIZE)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            base = 120 + 90 * math.sin(x * 0.02) * math.cos(y * 0.025)
            pixels[x, y] = max(0, min(255, int(base + random.gauss(0, 26))))
    image.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB").save(path)


def _as_jpeg(source_path, target_path, quality=72):
    """Re-encode so the set includes real compression artifacts."""
    Image.open(source_path).convert("RGB").save(target_path, "JPEG", quality=quality)


FIXTURES = (
    ("01-flat-shapes", flat_shapes),
    ("02-outlined-icon", outlined_icon),
    ("03-line-sketch", line_sketch),
    ("04-logo-mark", logo_mark),
    ("05-smooth-gradient", smooth_gradient),
    ("06-grain-texture", grain_texture),
)

# Fixtures also emitted as JPEG, to cover ringing around hard edges.
JPEG_VARIANTS = ("02-outlined-icon", "04-logo-mark")


def main(target):
    os.makedirs(target, exist_ok=True)
    written = []

    for name, builder in FIXTURES:
        path = os.path.join(target, f"{name}.png")
        builder(path)
        written.append(path)

    for name in JPEG_VARIANTS:
        source = os.path.join(target, f"{name}.png")
        jpeg = os.path.join(target, f"{name}-jpeg.jpg")
        _as_jpeg(source, jpeg)
        written.append(jpeg)

    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written)} fixtures in {target}")
    print("Drop your own images in the same folder - bench.py picks up everything there.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/bench")
