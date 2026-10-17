"""Connected pipe network: pipes are graph edges, every node gets a fitting (elbow / tee / reducer / flanged joint /
blind flange / equipment nozzle flange).  No pipe can end without a fitting; supports are generated for long runs."""
from __future__ import annotations

import math

import numpy as np

from ..geom import Q_IDENT
from . import equipment as EQ
from .common import Ctx, cube, cyl, qalign, sph


def _key(p):
    return tuple(int(round(float(c) * 1000)) for c in p)


class PipeNet:
    def __init__(self, ctx: Ctx, group: str):
        self.ctx, self.group = ctx, group
        ctx.kit.xform(group)
        self.nodes: dict = {}
        self.edges: list = []
        self.supports = 0
        self.links = []  # (node key, node key) joined through an inline valve
        self.fittings: dict = {}
        self._n = 0

    # ------------------------------------------------------------------ graph building
    def _node(self, p):
        k = _key(p)
        n = self.nodes.setdefault(k, dict(p=np.asarray(p, float), edges=[], equip=None))
        return n

    def equip(self, p, label="equipment"):
        self._node(p)["equip"] = label

    def pipe(self, a, b, r, mat="steel", bands=(), name=None, support=None, service=None):
        """Straight pipe between two nodes. support: None | 'down' | ('back', y_channel) | 'none' -> brackets every <=1.3 m
        along horizontal runs (to the deck, or sideways to a wall channel)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 0.02:
            return None
        u = (b - a) / L
        nm = name or f"p{self._n}"
        self._n += 1
        ctx = self.ctx
        mid = (a + b) / 2
        sh = [cyl(r, L, mat=mat)] + [cyl(r * 1.05, 0.14, (0, 0, (t - L / 2)), c, col=False) for t, c in bands if 0.07 < t < L - 0.07]
        ctx.static(f"{self.group}/{nm}", mid, qalign(u), sh, category="pipe_segment", radius=float(r))
        e = dict(name=nm, a=_key(a), b=_key(b), r=float(r), length=L, path=f"{self.group}/{nm}", mat=mat, service=service)
        self.edges.append(e)
        self._node(a)["edges"].append((u, float(r), nm))
        self._node(b)["edges"].append((-u, float(r), nm))
        # supports for horizontal runs
        if support not in (None, "none") and abs(u[2]) < 0.2 and L > 0.6:
            n = max(1, int(L / 1.3))
            for i in range(n):
                t = (i + 0.5) * L / n
                p = a + u * t
                self._support(p, r, support, f"{nm}_s{i}")
        return e

    def run(self, a, b, r, valves=(), mat="steel", bands=(), name=None, support=None, service=None, taps=()):
        """Pipe a->b split around inline valves = [(t_centre, face_to_face_length)] (metres from a). Valve ports become
        equipment nodes (flanged)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L
        cuts = sorted((t - ln / 2, t + ln / 2) for t, ln in valves)
        events = sorted([(c0, c1) for c0, c1 in cuts] + [(t, t) for t in taps])
        t0 = 0.0
        pts = []
        for c0, c1 in events:
            pts.append((t0, c0))
            if c1 > c0:
                self.equip(a + u * c0, "valve_port")
                self.equip(a + u * c1, "valve_port")
                self.links.append((_key(a + u * c0), _key(a + u * c1)))
            t0 = c1
        pts.append((t0, L))
        for i, (s0, s1) in enumerate(pts):
            if s1 - s0 < 0.02:
                continue
            bb = [(t - s0, c) for t, c in bands if s0 < t < s1]
            self.pipe(a + u * s0, a + u * s1, r, mat, bb, name=None if name is None else f"{name}_{i}", support=support, service=service)

    def _support(self, p, r, kind, name):
        ctx = self.ctx
        sh = [cyl(r * 1.12, 0.035, (0, 0, 0), "steel_dark", col=False, axis="x")]
        if kind == "down":
            h = p[2] - r
            if h > 0.06:
                ctx.static(f"{self.group}/{name}", (p[0], p[1], 0.0), Q_IDENT, [cyl(0.02, h, (0, 0, h / 2), "steel_dark"), cube((0.12, 0.12, 0.012), (0, 0, 0.006), "steel_dark", col=False)])
        elif isinstance(kind, tuple) and kind[0] == "up":  # hanger rod to the rack beam above
            h = kind[1] - (p[2] + r)
            if h > 0.05:
                ctx.static(f"{self.group}/{name}", (p[0], p[1], p[2] + r + h / 2), Q_IDENT, [cyl(0.012, h, (0, 0, 0), "steel_dark", col=False), cube((0.06, 0.06, 0.01), (0, 0, -h / 2 + 0.005), "steel_dark", col=False)])
        elif isinstance(kind, tuple) and kind[0] == "back":
            yc = kind[1]
            d = yc - (p[1] + r)
            if d > 0.01:
                ctx.static(f"{self.group}/{name}", (p[0], p[1] + r + d / 2, p[2]), Q_IDENT, [cube((0.04, d, 0.04), (0, 0, 0), "steel_dark"), cube((0.05, 0.03, 0.12), (0, d / 2, 0), "steel_dark", col=False)])
        else:
            return
        self.supports += 1

    # ------------------------------------------------------------------ fittings
    def finish(self):
        ctx = self.ctx
        counts = dict(elbow=0, tee=0, reducer=0, flanged_joint=0, blind_flange=0, equipment_flange=0, cross=0)
        for i, (k, n) in enumerate(self.nodes.items()):
            p, E = n["p"], n["edges"]
            if not E:
                continue
            rmax = max(e[1] for e in E)
            nm = f"{self.group}/fit{i}"
            if len(E) == 1:
                u, r, _ = E[0]
                t = "equipment_flange" if n["equip"] else "blind_flange"
                ctx.static(nm, p + (-u) * 0.0 + u * 0.0125, qalign(u), [cyl(r * 1.45, 0.025, mat="steel_c"), cyl(r * 1.1, 0.012, (0, 0, 0.018 if not n["equip"] else -0.0), "steel_dark", col=False)], category="fitting", kind=t)
                counts[t] += 1
            elif len(E) == 2:
                (u1, r1, _), (u2, r2, _) = E
                if float(np.dot(u1, u2)) < -0.97:  # collinear
                    ax = qalign(u1)
                    if abs(r1 - r2) > 0.002:
                        rm = (r1 + r2) / 2
                        ctx.static(nm, p, ax, [cyl(rm, 0.12, mat="steel"), cyl(max(r1, r2) * 1.4, 0.025, (0, 0, 0.06), "steel_c"), cyl(min(r1, r2) * 1.4, 0.025, (0, 0, -0.06), "steel_c")], category="fitting", kind="reducer")
                        counts["reducer"] += 1
                    else:
                        ctx.static(nm, p, ax, [cyl(r1 * 1.4, 0.03, mat="steel_c")], category="fitting", kind="flanged_joint")
                        counts["flanged_joint"] += 1
                else:
                    ctx.static(nm, p, Q_IDENT, [sph(rmax * 1.08, mat="steel"), cyl(rmax * 1.3, 0.02, tuple(E[0][0] * (rmax * 1.0 + 0.02)), "steel_c", col=False, quat=qalign(E[0][0]))], category="fitting", kind="elbow")
                    counts["elbow"] += 1
            else:
                kind = "tee" if len(E) == 3 else "cross"
                rb = min(e[1] for e in E)
                ctx.static(nm, p, Q_IDENT, [sph(max(rb * 1.5, rmax * 1.03), mat="steel")], category="fitting", kind=kind)
                counts[kind] += 1
        self.fittings = counts
        return counts

    # ------------------------------------------------------------------ report / validation
    def report(self):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self.edges:
            parent[find(e["a"])] = find(e["b"])
        for ka, kb in self.links:
            parent[find(ka)] = find(kb)
        comps = {}
        for e in self.edges:
            comps.setdefault(find(e["a"]), []).append(e["name"])
        dangling = [e["name"] for e in self.edges for k in (e["a"], e["b"]) if not self.nodes[k]["edges"]]
        diam = [2 * e["r"] for e in self.edges]
        return dict(n_pipe_segments=len(self.edges), n_nodes=len(self.nodes), fittings=self.fittings, n_supports=self.supports, n_connected_components=len(comps),
                    component_sizes=sorted((len(v) for v in comps.values()), reverse=True), dangling_pipe_ends_without_fitting=len(dangling),
                    diameter_range_m=[round(min(diam), 3), round(max(diam), 3)] if diam else None)
