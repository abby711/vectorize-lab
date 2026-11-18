"""SVG output for v2: one element per shape, primitives where recognised."""


def _number(value):
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _point(point, scale):
    return f"{_number(point[0] * scale)} {_number(point[1] * scale)}"


def path_data(loops, scale):
    """loops: list of segment lists, each closed."""
    parts = []
    for segments in loops:
        if not segments:
            continue
        parts.append(f"M{_point(segments[0].start, scale)}")
        for segment in segments:
            if segment.kind == "line":
                parts.append(f"L{_point(segment.end, scale)}")
            else:
                parts.append(
                    f"C{_point(segment.control_1, scale)} "
                    f"{_point(segment.control_2, scale)} {_point(segment.end, scale)}"
                )
        parts.append("Z")
    return "".join(parts)


def primitive_element(primitive, fill, scale):
    cx, cy = primitive.centre
    if primitive.kind == "circle":
        return (
            f'<circle cx="{_number(cx * scale)}" cy="{_number(cy * scale)}" '
            f'r="{_number(primitive.size[0] * scale)}" fill="{fill}"/>'
        )
    if primitive.kind == "ellipse":
        rotate = ""
        if primitive.angle:
            rotate = (
                f' transform="rotate({_number(primitive.angle)} '
                f'{_number(cx * scale)} {_number(cy * scale)})"'
            )
        return (
            f'<ellipse cx="{_number(cx * scale)}" cy="{_number(cy * scale)}" '
            f'rx="{_number(primitive.size[0] * scale)}" ry="{_number(primitive.size[1] * scale)}"'
            f'{rotate} fill="{fill}"/>'
        )
    width, height = primitive.size
    corner = f' rx="{_number(primitive.radius * scale)}"' if primitive.radius > 0 else ""
    return (
        f'<rect x="{_number((cx - width / 2) * scale)}" y="{_number((cy - height / 2) * scale)}" '
        f'width="{_number(width * scale)}" height="{_number(height * scale)}"{corner} fill="{fill}"/>'
    )


def gradient_element(gradient, ident, scale):
    stops = "".join(
        f'<stop offset="{_number(min(max(offset, 0.0), 1.0))}" stop-color="{to_hex(_rounded(colour))}"'
        + (f' stop-opacity="{_number(opacity)}"' if opacity < 0.995 else "")
        + "/>"
        for offset, colour, opacity in gradient.stops
    )
    if gradient.kind == "linear":
        x1, y1, x2, y2 = (_number(value * scale) for value in gradient.geometry)
        return (
            f'<linearGradient id="{ident}" gradientUnits="userSpaceOnUse" '
            f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">{stops}</linearGradient>'
        )
    cx, cy, radius = (_number(value * scale) for value in gradient.geometry)
    return (
        f'<radialGradient id="{ident}" gradientUnits="userSpaceOnUse" '
        f'cx="{cx}" cy="{cy}" r="{radius}">{stops}</radialGradient>'
    )


def glow_element(gradient, fill, scale):
    """The circle a fading radial gradient is drawn in."""
    cx, cy, radius = (_number(value * scale) for value in gradient.geometry)
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}"/>'


def _rounded(colour):
    return [min(255, max(0, round(channel))) for channel in colour]


def build_document(elements, width, height, defs=()):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    if defs:
        lines.append("<defs>")
        lines.extend(defs)
        lines.append("</defs>")
    lines.extend(elements)
    lines.append("</svg>")
    return "\n".join(lines)


def to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
