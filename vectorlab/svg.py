"""Stage 7: SVG emission."""


def _number(value):
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _point(point):
    return f"{_number(point[0])} {_number(point[1])}"


def path_data(rings_segments, scale):
    """rings_segments: list of (start_point, segments) for one shape."""
    parts = []
    for start_point, segments in rings_segments:
        parts.append(f"M{_point(start_point * scale)}")
        for segment in segments:
            if segment["kind"] == "line":
                parts.append(f"L{_point(segment['to'] * scale)}")
            else:
                parts.append(
                    "C"
                    f"{_point(segment['c1'] * scale)} "
                    f"{_point(segment['c2'] * scale)} "
                    f"{_point(segment['to'] * scale)}"
                )
        parts.append("Z")
    return "".join(parts)


def build_svg(shapes, width, height, background=None):
    """shapes: list of {"data": str, "fill": "#rrggbb"} in paint order."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    if background:
        lines.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    for shape in shapes:
        lines.append(
            f'<path d="{shape["data"]}" fill="{shape["fill"]}" fill-rule="evenodd"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
