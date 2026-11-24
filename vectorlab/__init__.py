"""vectorlab: tools for editable vector graphics, on one shared engine.

Each tool is its own module, with its own entry point, and loads only
what it needs:

    from vectorlab.trace import trace        # raster image -> SVG       command: vectorize

`vectorlab.core` is the shared engine and `vectorlab.v1` the first
version of the tracer, kept for comparison. The names below are also
available from here, loaded on first use.
"""

import importlib

__version__ = "0.2.0"

_EXPORTS = {
    "trace": ("vectorlab.trace", "trace"),
    "TraceOptions": ("vectorlab.trace", "TraceOptions"),
    "vectorize": ("vectorlab.trace", "trace"),
    "Options": ("vectorlab.trace", "TraceOptions"),
    "V2Options": ("vectorlab.trace", "TraceOptions"),
    "vectorize_v2": ("vectorlab.trace", "trace"),
    "VectorizeResult": ("vectorlab.core.result", "VectorizeResult"),
    "VectorizeRejected": ("vectorlab.core.result", "VectorizeRejected"),
    "VectorizeOptions": ("vectorlab.v1", "VectorizeOptions"),
    "vectorize_image": ("vectorlab.v1", "vectorize_image"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'vectorlab' has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    return getattr(importlib.import_module(module), attribute)
