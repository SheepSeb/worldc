"""Spawn interpenetration, floating bodies and hazard-path sweeps (generalised from static_validate / plant)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from pxr import Usd, UsdPhysics

from ..colliders import Collider, shrunk
from ..geom import OBB, hull_penetration, obb_penetration, qmat, qyaw

PEN_TOL = 0.003
ARTIC_PREFIX = ("door_", "gate_", "drawer", "hazard", "conveyor")


# --------------------------------------------------------------------------- joints
def _rigid_owner(stage, path: str) -> str:
    p = stage.GetPrimAtPath(path)
    q = p
    while q and q.GetPath().pathString != "/":
        if q.HasAPI(UsdPhysics.RigidBodyAPI):
            return q.GetPath().pathString
        q = q.GetParent()
    return path


def joint_adjacency(stage: Usd.Stage) -> dict[str, set[str]]:
    """prim-path node -> set of nodes it is jointed to (nodes are rigid-body paths, or the joint-target prim itself
    when that is a static prim)."""
    adj: dict[str, set[str]] = defaultdict(set)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        n0 = _rigid_owner(stage, t0[0].pathString) if t0 else None
        n1 = _rigid_owner(stage, t1[0].pathString) if t1 else None
        if n0:
            adj[n0]  # a body with any joint is "jointed", even to the world
        if n1:
            adj[n1]
        if n0 and n1:
            adj[n0].add(n1)
            adj[n1].add(n0)
    return adj


def _keys(c: Collider, adj) -> tuple:
    """Joint-graph nodes this collider belongs to: its rigid body, or (static) any ancestor prim that is a joint target."""
    if c.body:
        return (c.body,) if c.body in adj else ()
    out = []
    p = c.path
    while "/" in p and p != "":
        if p in adj:
            out.append(p)
        p = p.rsplit("/", 1)[0]
    return tuple(out)


# --------------------------------------------------------------------------- broadphase
def aabbs(cols):
    lo = np.array([c.obb.aabb()[0] for c in cols]).reshape(-1, 3)
    hi = np.array([c.obb.aabb()[1] for c in cols]).reshape(-1, 3)
    return lo, hi


def candidate_pairs(cols: list[Collider], lo, hi):
    """Vectorised AABB broadphase, pre-filtered for the generic skip rules (mesh, static-static, same body/group,
    hazard-hazard)."""
    n = len(cols)
    if n == 0:
        return []
    bid = {}
    body = np.array([bid.setdefault(c.body, len(bid)) if c.body else -1 for c in cols])
    gid = {}
    grp = np.array([gid.setdefault(c.group, len(gid) + 1) if c.group else 0 for c in cols])
    mesh = np.array([c.is_mesh for c in cols])
    haz = np.array([c.category == "hazard" for c in cols])
    pairs = []
    for s in range(0, n, 256):
        e = min(n, s + 256)
        ov = np.all((lo[s:e, None, :] < hi[None, :, :]) & (hi[s:e, None, :] > lo[None, :, :]), axis=2)
        ii = np.arange(s, e)[:, None]
        jj = np.arange(n)[None, :]
        ok = ov & (jj > ii)
        ok &= ~(mesh[s:e, None] | mesh[None, :])
        ok &= ~((body[s:e, None] < 0) & (body[None, :] < 0))
        ok &= ~((body[s:e, None] == body[None, :]) & (body[s:e, None] >= 0))
        ok &= ~((grp[s:e, None] == grp[None, :]) & (grp[s:e, None] > 0))
        ok &= ~(haz[s:e, None] & haz[None, :])
        a, b = np.nonzero(ok)
        pairs.extend(zip((a + s).tolist(), b.tolist()))
    return pairs


def spawn_penetrations(cols, adj, lo, hi):
    keys = [_keys(c, adj) for c in cols]
    out = []
    for a, b in candidate_pairs(cols, lo, hi):
        ka, kb = keys[a], keys[b]
        if ka and kb and any(y in adj.get(x, ()) for x in ka for y in kb):
            continue
        A, B = cols[a], cols[b]
        pen = obb_penetration(shrunk(A), shrunk(B))
        if pen > PEN_TOL and (A.points is not None or B.points is not None):  # a hull's box over-reports: test the hull itself
            pa = A.points if A.points is not None else shrunk(A).corners()
            pb = B.points if B.points is not None else shrunk(B).corners()
            pen = min(pen, hull_penetration(pa, pb, [A.obb.R.T, B.obb.R.T]))
        if pen > PEN_TOL:
            out.append((a, b, pen))
    return out


# --------------------------------------------------------------------------- floating
def floating_bodies(cols, adj, lo, hi):
    by_body = defaultdict(list)
    for i, c in enumerate(cols):
        if c.body:
            by_body[c.body].append(i)
    mesh = np.array([c.is_mesh for c in cols])
    bodies = np.array([c.body or "" for c in cols])
    out = []
    for body, idx in by_body.items():
        c0 = cols[idx[0]]
        if c0.kinematic or c0.group or c0.category.startswith(ARTIC_PREFIX) or body in adj:
            continue
        zb = lo[idx, 2].min()
        if zb < 0.02:
            continue
        mn, mx = lo[idx].min(0), hi[idx].max(0)
        near_top = np.abs(hi[:, 2] - zb) <= 0.012
        straddle = (lo[:, 2] < zb) & (zb < hi[:, 2])
        xy = (hi[:, 0] > mn[0] + 0.01) & (lo[:, 0] < mx[0] - 0.01) & (hi[:, 1] > mn[1] + 0.01) & (lo[:, 1] < mx[1] - 0.01)
        sup = (near_top | straddle) & xy & ~mesh & (bodies != body)
        if not sup.any():
            out.append((idx[0], body, float(zb)))
    return out


# --------------------------------------------------------------------------- hazard sweep
def hazard_conflicts(cols, hazards, lo, hi, step=0.2):
    out = []
    static_idx = np.array([i for i, c in enumerate(cols) if not c.is_mesh and c.category != "hazard"], int)
    for h in hazards:
        hb = [c for c in cols if c.body == h["path"]]
        pts = [list(map(float, p[:2])) for p in h.get("waypoints", [])]
        if len(pts) < 2 or not hb:
            continue
        if h.get("path_mode", "loop") == "loop":
            pts = pts + [pts[0]]
        samples = []
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            L = math.hypot(bx - ax, by - ay)
            if L < 1e-9:
                continue
            yaw = math.atan2(by - ay, bx - ax)
            for s in np.arange(0.0, L, step):
                samples.append((ax + (bx - ax) * s / L, ay + (by - ay) * s / L, yaw))
        worst, hit = 0.0, None
        for (x, y, yaw) in samples:
            for c in hb:
                base = shrunk(c)
                o = OBB(np.array([x, y, c.obb.c[2]]), base.h.copy(), qmat(qyaw(yaw)))
                olo, ohi = o.aabb()
                m = np.all(olo < hi[static_idx], axis=1) & np.all(ohi > lo[static_idx], axis=1)
                for j in static_idx[m]:
                    pen = obb_penetration(o, shrunk(cols[j]))
                    if pen > worst:
                        worst, hit = pen, int(j)
        if worst > PEN_TOL:
            out.append((h["path"], hit, worst))
    return out
