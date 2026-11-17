#!/usr/bin/env python3
"""Seeded generator for the complex ground-truth fixtures.

Each scene targets one way tracers break, so a regression points at a
cause rather than an average:

    20-city-night     hundreds of tiny shapes: windows, stars
    21-confetti       heavy overlap, colours reused across shapes
    22-icon-grid      small-scale detail: 24 icons at ~60 px
    23-gradient-sky   gradients — a known weakness, kept honest
    24-line-art       strokes rather than fills — a known weakness
    25-scene-tiny     a familiar scene rendered at 160 px wide
    26-badge-large    a familiar scene rendered at 1440 px wide

    python3 dataset/generate.py        # writes dataset/svg/2*.svg
"""

import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "svg")


def _svg(width, height, body, raster_width=None):
    extra = f' data-raster-width="{raster_width}"' if raster_width else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}"{extra}>\n' + "\n".join(body) + "\n</svg>\n"
    )


def city_night(rng):
    width, height = 640, 400
    body = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#141b33"/>']
    body.append('<circle cx="520" cy="80" r="34" fill="#f2e8c9"/>')
    body.append('<circle cx="534" cy="70" r="30" fill="#141b33"/>')
    for _ in range(40):
        x, y = rng.uniform(10, width - 10), rng.uniform(10, 170)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rng.choice([1.5, 2, 2.5]):.1f}" fill="#dfe6ff"/>')

    layers = (("#232c4f", 140, 230), ("#2f3a63", 180, 300), ("#3d4a7a", 230, 340))
    for colour, lowest, highest in layers:
        x = -10
        while x < width:
            building = rng.uniform(34, 70)
            top = height - rng.uniform(lowest, highest) * 0.55 - 60
            body.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{building:.1f}" height="{height - top:.1f}" fill="{colour}"/>'
            )
            if colour == "#3d4a7a":
                for row in range(int((height - top - 20) // 16)):
                    for column in range(int((building - 8) // 12)):
                        if rng.random() < 0.45:
                            wx, wy = x + 6 + column * 12, top + 10 + row * 16
                            body.append(
                                f'<rect x="{wx:.1f}" y="{wy:.1f}" width="6" height="8" fill="#f6c95c"/>'
                            )
            x += building + rng.uniform(2, 8)
    body.append(f'<rect x="0" y="{height - 26}" width="{width}" height="26" fill="#0d1226"/>')
    return _svg(width, height, body)


def confetti(rng):
    width, height = 480, 360
    colours = ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c", "#f78c6b", "#9b5de5", "#fefae0"]
    body = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff8ee"/>']
    for _ in range(60):
        colour = rng.choice(colours)
        cx, cy = rng.uniform(20, width - 20), rng.uniform(20, height - 20)
        size = rng.uniform(10, 38)
        kind = rng.choice(["circle", "rect", "triangle", "pill"])
        angle = rng.uniform(0, 180)
        if kind == "circle":
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size:.1f}" fill="{colour}"/>')
        elif kind == "rect":
            body.append(
                f'<rect x="{cx - size:.1f}" y="{cy - size * 0.6:.1f}" width="{2 * size:.1f}" '
                f'height="{1.2 * size:.1f}" rx="{size * 0.2:.1f}" fill="{colour}" '
                f'transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"/>'
            )
        elif kind == "pill":
            body.append(
                f'<rect x="{cx - size:.1f}" y="{cy - size * 0.3:.1f}" width="{2 * size:.1f}" '
                f'height="{0.6 * size:.1f}" rx="{size * 0.3:.1f}" fill="{colour}" '
                f'transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"/>'
            )
        else:
            points = " ".join(
                f"{cx + size * math.cos(math.radians(angle + 120 * index)):.1f},"
                f"{cy + size * math.sin(math.radians(angle + 120 * index)):.1f}"
                for index in range(3)
            )
            body.append(f'<polygon points="{points}" fill="{colour}"/>')
    return _svg(width, height, body)


def _icon(kind, x, y, size, ink):
    centre_x, centre_y = x + size / 2, y + size / 2
    r = size * 0.28
    if kind == 0:
        return [f'<circle cx="{centre_x}" cy="{centre_y}" r="{r}" fill="{ink}"/>']
    if kind == 1:
        return [
            f'<circle cx="{centre_x}" cy="{centre_y}" r="{r}" fill="{ink}"/>',
            f'<circle cx="{centre_x}" cy="{centre_y}" r="{r * 0.55}" fill="#ffffff"/>',
        ]
    if kind == 2:
        return [
            f'<rect x="{centre_x - r}" y="{centre_y - r * 0.18}" width="{2 * r}" height="{r * 0.36}" rx="2" fill="{ink}"/>',
            f'<rect x="{centre_x - r * 0.18}" y="{centre_y - r}" width="{r * 0.36}" height="{2 * r}" rx="2" fill="{ink}"/>',
        ]
    if kind == 3:
        return [
            f'<polygon points="{centre_x},{centre_y - r} {centre_x + r},{centre_y + r * 0.8} '
            f'{centre_x - r},{centre_y + r * 0.8}" fill="{ink}"/>'
        ]
    if kind == 4:
        return [
            f'<rect x="{centre_x - r}" y="{centre_y - r * 0.7 + index * r * 0.55}" width="{2 * r * (1 - 0.25 * index)}" '
            f'height="{r * 0.3}" rx="{r * 0.15}" fill="{ink}"/>'
            for index in range(3)
        ]
    return [
        f'<path d="M{centre_x} {centre_y + r} C{centre_x - r * 1.6} {centre_y - r * 0.3} '
        f'{centre_x - r * 0.5} {centre_y - r * 1.3} {centre_x} {centre_y - r * 0.4} '
        f'C{centre_x + r * 0.5} {centre_y - r * 1.3} {centre_x + r * 1.6} {centre_y - r * 0.3} '
        f'{centre_x} {centre_y + r} Z" fill="{ink}"/>'
    ]


def icon_grid(rng):
    width, height = 640, 480
    tiles = ["#e8f1ff", "#fff1e6", "#e9f9ee", "#f6ebff"]
    inks = ["#2357d9", "#e0621c", "#1e9e57", "#7a3fd1"]
    body = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    size, gap = 76, 24
    for row in range(4):
        for column in range(6):
            x = 40 + column * (size + gap) - 8
            y = 40 + row * (size + gap) + 8
            tone = (row + column) % 4
            body.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="16" fill="{tiles[tone]}"/>')
            body.extend(_icon((row * 6 + column) % 6, x, y, size, inks[tone]))
    return _svg(width, height, body)


def gradient_sky(rng):
    width, height = 480, 360
    body = [
        "<defs>",
        '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#1b2a6b"/><stop offset="0.6" stop-color="#e86f68"/>'
        '<stop offset="1" stop-color="#f7c873"/></linearGradient>',
        '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
        '<stop offset="0" stop-color="#fff4c2"/><stop offset="1" stop-color="#ffd27a" stop-opacity="0"/>'
        "</radialGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#sky)"/>',
        '<circle cx="240" cy="230" r="90" fill="url(#glow)"/>',
        '<circle cx="240" cy="230" r="38" fill="#fff4c2"/>',
        '<path d="M0 270 C90 230 170 260 240 252 C320 244 390 220 480 246 L480 360 L0 360 Z" fill="#3a2f5b"/>',
        '<path d="M0 310 C110 284 220 318 320 300 C390 288 440 296 480 304 L480 360 L0 360 Z" fill="#241c3d"/>',
    ]
    return _svg(width, height, body)


def line_art(rng):
    width, height = 480, 360
    stroke = 'fill="none" stroke="#1d2433" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"'
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbf7ef"/>',
        f'<path d="M110 300 L110 190 L190 130 L270 190 L270 300 Z" {stroke}/>',
        f'<path d="M168 300 L168 240 L212 240 L212 300" {stroke}/>',
        f'<path d="M128 206 L156 206 L156 232 L128 232 Z" {stroke}/>',
        f'<circle cx="380" cy="90" r="30" {stroke}/>',
        f'<path d="M360 300 L360 230 M360 250 C330 240 320 210 340 190 C350 214 362 226 360 250 '
        f'M360 236 C392 226 404 196 386 174 C372 198 360 212 360 236" {stroke}/>',
        f'<path d="M40 300 L440 300" {stroke}/>',
    ]
    return _svg(width, height, body)


def _reuse(name, raster_width):
    with open(os.path.join(OUT, name), encoding="utf-8") as handle:
        text = handle.read()
    return text.replace("<svg ", f'<svg data-raster-width="{raster_width}" ', 1)


def main():
    rng = random.Random(2026)
    scenes = {
        "20-city-night.svg": city_night(rng),
        "21-confetti.svg": confetti(rng),
        "22-icon-grid.svg": icon_grid(rng),
        "23-gradient-sky.svg": gradient_sky(rng),
        "24-line-art.svg": line_art(rng),
        "25-scene-tiny.svg": _reuse("10-flat-scene.svg", 160),
        "26-badge-large.svg": _reuse("12-badge.svg", 1440),
    }
    for name, text in scenes.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote dataset/svg/{name}")


if __name__ == "__main__":
    main()
