"""Counts editable points in any SVG, the same way for every engine.

The unit is the anchor a designer would drag. Control handles don't
count. A closed subpath that returns to its start counts that point
once. Primitives count as the anchors they'd show in an editor: a
circle, ellipse or rect is 4, a line is 2, a polygon is its vertex count.
"""

import re

_ARG_COUNTS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
_TOKEN = re.compile(r"[MLHVCSQTAZmlhvcsqtaz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PRIMITIVE_COST = {"circle": 4, "ellipse": 4, "rect": 4, "line": 2}


def _subpath_anchors(data):
    """Yields the anchor count of each subpath in a path's d attribute."""
    tokens = _TOKEN.findall(data)
    position = 0
    command = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    anchors = []

    def flush():
        if not anchors:
            return 0
        count = len(anchors)
        if count > 1 and abs(anchors[-1][0] - anchors[0][0]) < 1e-6 and abs(
            anchors[-1][1] - anchors[0][1]
        ) < 1e-6:
            count -= 1
        return count

    while position < len(tokens):
        token = tokens[position]
        if token.isalpha():
            command = token
            position += 1
            if command in "Zz":
                current = start
                yield flush()
                anchors = []
            continue

        upper = command.upper()
        relative = command.islower()
        arity = _ARG_COUNTS.get(upper, 0)
        if arity == 0 or position + arity > len(tokens):
            break

        values = [float(value) for value in tokens[position : position + arity]]
        position += arity

        if upper == "H":
            x = values[0] + (current[0] if relative else 0.0)
            point = (x, current[1])
        elif upper == "V":
            y = values[0] + (current[1] if relative else 0.0)
            point = (current[0], y)
        else:
            x, y = values[-2], values[-1]
            if relative:
                x += current[0]
                y += current[1]
            point = (x, y)

        if upper == "M":
            yield flush()
            anchors = [point]
            start = point
            command = "l" if relative else "L"
        else:
            anchors.append(point)
        current = point

    yield flush()


def path_anchor_count(data):
    return sum(_subpath_anchors(data))


_NOT_RENDERED = {
    "defs", "symbol", "clipPath", "mask", "pattern", "marker", "linearGradient",
    "radialGradient", "title", "desc", "metadata", "style", "script", "filter",
}


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _element_cost(element):
    """(anchors, is_path) for a single shape element, or None if it isn't one."""
    name = _local(element.tag)
    if name == "path":
        return path_anchor_count(element.get("d", "")), True
    if name in _PRIMITIVE_COST:
        return _PRIMITIVE_COST[name], False
    if name in ("polygon", "polyline"):
        return len(re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", element.get("points", ""))) // 2, False
    return None


def _features(element, features):
    style = (element.get("style") or "").replace(" ", "")
    stroke = element.get("stroke") or ""
    if (stroke and stroke != "none") or ("stroke:" in style and "stroke:none" not in style):
        features.add("strokes")
    for value in (element.get("fill") or "", style):
        if "url(#" in value:
            features.add("gradient-or-pattern")
    name = _local(element.tag)
    if name in ("text", "tspan"):
        features.add("text")
    if name == "image":
        features.add("embedded-image")
    if element.get("clip-path") or element.get("mask"):
        features.add("clipping")
    if name == "use":
        features.add("reuse")


def _walk(element, ids, totals, features, depth=0):
    if depth > 32:
        return
    name = _local(element.tag)
    if name in _NOT_RENDERED:
        return
    _features(element, features)
    if name == "use":
        reference = element.get("href") or element.get("{http://www.w3.org/1999/xlink}href") or ""
        target = ids.get(reference.lstrip("#"))
        if target is not None:
            # A reused shape is drawn again, so a tracer must redraw it too.
            _walk_target(target, ids, totals, features, depth + 1)
        return
    cost = _element_cost(element)
    if cost is not None:
        anchors, is_path = cost
        totals["anchors"] += anchors
        totals["paths" if is_path else "primitives"] += 1
    for child in element:
        _walk(child, ids, totals, features, depth + 1)


def _walk_target(target, ids, totals, features, depth):
    """A <use> target, which may itself live inside <defs> or be a <symbol>."""
    name = _local(target.tag)
    if name in ("symbol", "defs", "g"):
        for child in target:
            _walk(child, ids, totals, features, depth + 1)
    else:
        _walk(target, ids, totals, features, depth)


def analyse_svg(svg_text):
    """Anchors, shapes and features of an SVG, following <use> references.

    Returns None if the document can't be parsed safely.
    """
    import xml.etree.ElementTree as ElementTree

    if "<!ENTITY" in svg_text[:8192]:
        return None  # entity expansion is never needed for artwork, and can be an attack
    try:
        root = ElementTree.fromstring(svg_text.encode("utf-8"))
    except ElementTree.ParseError:
        return None
    ids = {element.get("id"): element for element in root.iter() if element.get("id")}
    totals = {"anchors": 0, "paths": 0, "primitives": 0}
    features = set()
    _walk(root, ids, totals, features)
    totals["shapes"] = totals["paths"] + totals["primitives"]
    totals["features"] = sorted(features)
    return totals


def count_editable_points(svg_text):
    """Returns {"shapes", "anchors", "paths", "primitives"} for an SVG document."""
    parsed = analyse_svg(svg_text)
    if parsed is not None:
        return {key: parsed[key] for key in ("shapes", "anchors", "paths", "primitives")}
    paths = re.findall(r'<path\b[^>]*?\bd="([^"]*)"', svg_text)
    anchors = sum(path_anchor_count(data) for data in paths)

    primitives = 0
    for tag, cost in _PRIMITIVE_COST.items():
        found = len(re.findall(rf"<{tag}\b", svg_text))
        primitives += found
        anchors += found * cost

    for points in re.findall(r'<poly(?:gon|line)\b[^>]*?\bpoints="([^"]*)"', svg_text):
        vertices = len(re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", points)) // 2
        primitives += 1
        anchors += vertices

    return {
        "shapes": len(paths) + primitives,
        "anchors": anchors,
        "paths": len(paths),
        "primitives": primitives,
    }
