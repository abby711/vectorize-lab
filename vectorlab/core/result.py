"""What every tool returns, and how it refuses input it can't handle."""

from dataclasses import dataclass


class VectorizeRejected(Exception):
    """Raised when the input is beyond what region tracing can represent.

    Failing loudly beats grinding for minutes and emitting a 20 MB file
    of noise: the caller can then fall back, or tell the user plainly.
    """

    def __init__(self, reason, region_count=0):
        super().__init__(reason)
        self.reason = reason
        self.region_count = region_count


@dataclass
class VectorizeResult:
    svg: str
    width: int
    height: int
    shape_count: int
    path_segment_count: int
    palette: list
    notes: tuple = ()
