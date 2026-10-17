"""Exact (grid-independent) narrowest-gap measurement between collider footprints."""
from __future__ import annotations

import numpy as np

from ..geom import largest_gap, scan_interval


def measure(polys: list[np.ndarray], boxes: list[tuple], check: dict, step=0.01):
    """Narrowest free gap along ``check['axis']``, minimised over scanlines across ``span``.

    ``polys`` are footprint polygons (xy) and ``boxes`` their (xmin, ymin, xmax, ymax). Returns (gap, scan coord)."""
    x0, y0, x1, y1 = check["rect"]
    axis = check.get("axis", "x")
    span = check.get("span")
    if axis == "x":
        a0, a1, s0, s1 = x0, x1, y0, y1  # gap along x; scan over y
        P = polys
        B = boxes
    else:
        a0, a1, s0, s1 = y0, y1, x0, x1
        P = [p[:, ::-1] for p in polys]  # swap so the gap axis is "x"
        B = [(b[1], b[0], b[3], b[2]) for b in boxes]
    if span:
        s0, s1 = span
    sel = [p for p, b in zip(P, B) if not (b[2] < a0 - 0.6 or b[0] > a1 + 0.6 or b[3] < s0 or b[1] > s1)]
    if not sel:
        return 0.0, float(s0)
    slo = np.array([p[:, 1].min() for p in sel])
    shi = np.array([p[:, 1].max() for p in sel])
    side = check.get("side", "any")
    worst, at = 1e9, None
    for s in np.arange(s0, s1, step):
        iv = [i for i in (scan_interval(sel[k], s) for k in np.nonzero((slo <= s + 1e-9) & (shi >= s - 1e-9))[0]) if i]
        if side == "max":  # blockage's +edge to the first obstacle beyond it (cf. solver)
            edge = max([b for a, b in iv if a < a1 - 0.3], default=a0)
            g = min([a for a, b in iv if a >= edge - 1e-9 and a > a1 - 0.3], default=a1) - edge
        elif side == "min":
            edge = min([a for a, b in iv if b > a0 + 0.3], default=a1)
            g = edge - max([b for a, b in iv if b <= edge + 1e-9 and b < a0 + 0.3], default=a0)
        else:
            g = largest_gap(iv)
        if g < worst:
            worst, at = g, float(s)
    if at is None:
        return 0.0, float(s0)
    return float(worst), at
