"""vectorlab: raster images (PNG, JPG) to editable SVG.

    from vectorlab import vectorize
    result = vectorize("logo.png")
    open("logo.svg", "w").write(result.svg)

`vectorize` is the current engine (vectorlab.v2); `vectorize_image` is the
first version, kept for comparison. Only numpy and Pillow are needed.
"""

from .pipeline import VectorizeOptions, VectorizeRejected, VectorizeResult, vectorize_image
from .v2 import V2Options, vectorize_v2

vectorize = vectorize_v2
Options = V2Options

__version__ = "0.2.0"

__all__ = [
    "Options",
    "V2Options",
    "VectorizeOptions",
    "VectorizeRejected",
    "VectorizeResult",
    "vectorize",
    "vectorize_image",
    "vectorize_v2",
]
