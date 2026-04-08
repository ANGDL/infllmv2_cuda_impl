"""Optional CUDA extension loader.

This module centralizes loading of the compiled extension so Python fallbacks
can be used when the extension is unavailable.
"""

from __future__ import annotations

try:
    from . import C as C  # type: ignore
except Exception:
    C = None
