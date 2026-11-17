"""vectorlab - a small raster-to-vector tracer.

Dependency-light on purpose: numpy + Pillow only, so every stage maps
cleanly onto a later C++17 port.
"""

from .pipeline import VectorizeOptions, VectorizeRejected, vectorize_image

__all__ = ["VectorizeOptions", "VectorizeRejected", "vectorize_image"]
