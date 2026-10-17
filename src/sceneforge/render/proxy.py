"""Fast software "proxy" renderer: flat-shaded polygons via Pillow, painter's algorithm, numpy-vectorised geometry.

Meant for layout inspection (LLM judge / MCP ``render``), not realism.  Core deps only (pxr, numpy, scipy, Pillow).

``render_proxy(scene, camera)``      perspective / orthographic view from a spec camera, USD camera prim or dict.
``render_topdown_map(scene)``        orthographic semantic map (collider footprints by category, zones, paths, goals).
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

NAMESPACES = ("scene", "warehouse", "plant")
GRAPH_NAME = "scene_graph.json"
SSAA = 2
MAX_MESH_FACES = 2000
KEY_LIGHT = np.array([0.40, -0.35, 0.85]) / np.linalg.norm([0.40, -0.35, 0.85])

CATEGORY_COLORS = {
    "wall": (0.62, 0.62, 0.64), "floor": (0.55, 0.55, 0.55), "roof": (0.5, 0.5, 0.52), "building": (0.62, 0.62, 0.64),
    "rack": (0.2, 0.35, 0.75), "rack_row": (0.2, 0.35, 0.75), "pallet": (0.72, 0.55, 0.3), "box": (0.75, 0.6, 0.38),
    "cardboard_box": (0.75, 0.6, 0.38), "tote": (0.2, 0.55, 0.3), "conveyor": (0.35, 0.35, 0.4), "door": (0.55, 0.4, 0.25),
    "human": (0.9, 0.35, 0.3), "forklift": (0.95, 0.75, 0.1), "robot": (0.1, 0.7, 0.8), "cage": (0.45, 0.45, 0.5),
    "camera": (1, 1, 1), "light": (1, 1, 0.8),
}
GREY = (0.6, 0.6, 0.6)


# ============================================================================ helpers
def _category_color(cat: str):
    if not cat:
        return GREY
    if cat in CATEGORY_COLORS:
        return CATEGORY_COLORS[cat]
    h = int(hashlib.md5(cat.encode()).hexdigest()[:6], 16)
    return colorsys.hsv_to_rgb((h % 360) / 360.0, 0.55, 0.85)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # very old Pillow
        return ImageFont.load_default()


def _load_scene(scene) -> tuple[Path, dict, Path]:
    """-> (scene_dir, scene_graph (may be {}), usd path)."""
    p = Path(scene)
    if p.is_dir():
        g = json.loads((p / GRAPH_NAME).read_text())
        return p, g, p / g["usd"]
    g = {}
    if (p.parent / GRAPH_NAME).exists():
        try:
            g = json.loads((p.parent / GRAPH_NAME).read_text())
        except Exception:
            g = {}
    return p.parent, g, p


def _prim_category(prim) -> str:
    p = prim
    while p and p.GetPath().pathString != "/":
        for ns in NAMESPACES:
            a = p.GetAttribute(f"{ns}:category")
            if a and a.HasValue():
                return str(a.Get())
        p = p.GetParent()
    return ""


def _make_hide(hide, extra_roof: bool):
    """Predicate (prim, path_str) -> bool.  Items starting with '/' are path prefixes; others match a path component
    (case-insensitive) or the prim category."""
    prefixes, names = [], set()
    for h in hide or []:
        h = str(h)
        if h.startswith("/"):
            prefixes.append(h.rstrip("/"))
        else:
            names.add(h.lower())

    def fn(prim, path: str) -> bool:
        if extra_roof and "/Roof" in path:
            return True
        for pre in prefixes:
            if path == pre or path.startswith(pre + "/"):
                return True
        if names:
            if any(c.lower() in names for c in path.split("/")):
                return True
            if _prim_category(prim).lower() in names:
                return True
        return False

    return fn


# ============================================================================ materials
class _Colors:
    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.mat: dict[str, tuple | None] = {}
        self.tex: dict[str, tuple | None] = {}

    def _texture_mean(self, tex_input) -> tuple | None:
        v = tex_input.Get()
        if v is None:
            return None
        path = getattr(v, "resolvedPath", "") or getattr(v, "path", "")
        if not path:
            return None
        if path in self.tex:
            return self.tex[path]
        res = None
        try:
            cand = Path(path)
            if not cand.exists():
                layer = tex_input.GetAttr().GetPropertyStack(Usd.TimeCode.Default())[0].layer
                cand = Path(Sdf.ComputeAssetPathRelativeToLayer(layer, getattr(v, "path", path)))
            with Image.open(cand) as im:
                im = im.convert("RGB")
                im.thumbnail((32, 32))
                res = tuple(float(c) / 255.0 for c in np.asarray(im, float).reshape(-1, 3).mean(0))
        except Exception:
            res = None
        self.tex[path] = res
        return res

    def _from_material(self, mat) -> tuple | None:
        key = mat.GetPath().pathString
        if key in self.mat:
            return self.mat[key]
        out = None
        try:
            sh = mat.ComputeSurfaceSource()[0]
            if sh:
                opacity = 1.0
                oi = sh.GetInput("opacity")
                if oi and oi.Get() is not None and not oi.HasConnectedSource():
                    opacity = float(oi.Get())
                di = sh.GetInput("diffuseColor")
                rgb = None
                if di:
                    if di.HasConnectedSource():
                        src = di.GetConnectedSource()
                        if src:
                            tin = UsdShade.Shader(src[0].GetPrim()).GetInput("file")
                            if tin:
                                rgb = self._texture_mean(tin)
                    elif di.Get() is not None:
                        rgb = tuple(float(c) for c in di.Get())
                if rgb is not None:
                    out = (rgb, opacity)
        except Exception:
            out = None
        self.mat[key] = out
        return out

    def get(self, prim):
        """-> (rgb 0..1, opacity)"""
        try:
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        except Exception:
            mat = None
        if mat:
            r = self._from_material(mat)
            if r:
                return r
        try:
            dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if dc is not None and len(dc):
                return tuple(float(c) for c in dc[0]), 1.0
        except Exception:
            pass
        return _category_color(_prim_category(prim)), 1.0



# ============================================================================ templates
def _unit_cube():
    v = np.array([[x, y, z] for z in (-.5, .5) for y in (-.5, .5) for x in (-.5, .5)], float)
    # index = x + 2y + 4z ; CCW seen from outside
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return v, np.array(f)


def _unit_prism(n: int, apex: bool = False):
    """Unit-radius, height-1 (z in -.5..+.5) n-gon prism (or cone with apex at +z).  Returns verts and polygon list."""
    a = np.arange(n) * 2 * np.pi / n
    bot = np.c_[np.cos(a), np.sin(a), -0.5 * np.ones(n)]
    if apex:
        v = np.vstack([bot, [[0, 0, 0.5]]])
        faces = [[i, (i + 1) % n, n, n] for i in range(n)] + [list(range(n - 1, -1, -1))]
        return v, faces
    top = bot.copy()
    top[:, 2] = 0.5
    v = np.vstack([bot, top])
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    faces.append(list(range(n - 1, -1, -1)))
    faces.append([n + i for i in range(n)])
    return v, faces


def _unit_icosphere():
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1),
         (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    cache, nf = {}, []

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            m = v[a] + v[b]
            v.append(m / np.linalg.norm(m))
            cache[k] = len(v) - 1
        return cache[k]

    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(v), [list(x) for x in nf]


_CUBE = _unit_cube()
_ICO = _unit_icosphere()


def _pad_faces(faces, K):
    arr = np.empty((len(faces), K), int)
    for i, f in enumerate(faces):
        arr[i, :len(f)] = f
        arr[i, len(f):] = f[-1]
    return arr


# ============================================================================ geometry gather
class _Polys:
    """Accumulates world-space polygons in two groups: K=4 (quads/triangles, padded) and K=12 (n-gons, padded)."""

    def __init__(self):
        self.parts = {4: [], 12: []}  # lists of (V(P,K,3), colour idx(P,), flip(P,), two(P,))
        self.colors: list[tuple] = []  # (rgb, opacity)
        self._cidx: dict[tuple, int] = {}
        self.n_prims = 0

    def cidx(self, rgb, op) -> int:
        k = (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3), round(op, 2))
        if k not in self._cidx:
            self._cidx[k] = len(self.colors)
            self.colors.append((rgb, op))
        return self._cidx[k]

    def add(self, V, ci, flip, two):
        K = V.shape[1]
        n = V.shape[0]
        self.parts[K].append((V, np.full(n, ci), np.full(n, flip), np.full(n, two)))

    def add_batch(self, V, ci, flip, two):
        self.parts[V.shape[1]].append((V, ci, flip, two))

    def cat(self, K):
        p = self.parts[K]
        if not p:
            return None
        return tuple(np.concatenate([x[i] for x in p]) for i in range(4))


def _mesh_polys(prim, M, ci, out: _Polys):
    m = UsdGeom.Mesh(prim)
    pts = m.GetPointsAttr().Get()
    cnt = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if pts is None or cnt is None or idx is None or len(cnt) == 0:
        return
    pts = np.asarray(pts, float)
    cnt = np.asarray(cnt, int)
    idx = np.asarray(idx, int)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    flip = -1.0 if np.linalg.det(M[:3, :3]) < 0 else 1.0
    if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        flip = -flip
    too_many = len(cnt) > MAX_MESH_FACES
    if all(c in (3, 4) for c in np.unique(cnt)) and not too_many:
        for k in (3, 4):
            sel = cnt == k
            if sel.any():
                ii = idx[off[sel][:, None] + np.arange(k)]
                if k == 3:
                    ii = np.concatenate([ii, ii[:, 2:3]], 1)
                V = pts[ii]
                out.add(_world(V, M), ci, flip, True)
        return
    # general: fan triangulation
    tris = []
    for o, c in zip(off, cnt):
        for j in range(1, c - 1):
            tris.append((idx[o], idx[o + j], idx[o + j + 1]))
    tris = np.array(tris, int)
    if len(tris) > MAX_MESH_FACES:
        pts, tris = _cluster_decimate(pts, tris)
    if len(tris) == 0:
        return
    ii = np.concatenate([tris, tris[:, 2:3]], 1)
    out.add(_world(pts[ii], M), ci, flip, True)


def _cluster_decimate(pts, tris):
    lo, hi = pts.min(0), pts.max(0)
    ext = max(float((hi - lo).max()), 1e-9)
    g = math.sqrt(MAX_MESH_FACES) * 0.9
    for _ in range(12):
        q = np.floor((pts - lo) / ext * g).astype(np.int64)
        key = (q[:, 0] * 100003 + q[:, 1]) * 100003 + q[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        cen = np.zeros((len(uniq), 3))
        np.add.at(cen, inv, pts)
        cen /= np.bincount(inv, minlength=len(uniq))[:, None]
        t = inv[tris]
        ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
        t = t[ok]
        if len(t) <= MAX_MESH_FACES:
            return cen, t
        g *= 0.75
    return cen, t[:MAX_MESH_FACES]


def _world(V, M):
    """Row-vector convention world transform of (...,3) points."""
    return V @ M[:3, :3] + M[3, :3]


def _capsule_polys(r, h, axis, n=10):
    """Returns (verts, faces) of a capsule along Z (cylinder part length h)."""
    rings = [(-h / 2 - r, 0.0), (-h / 2 - r * 0.7071, r * 0.7071), (-h / 2, r), (h / 2, r), (h / 2 + r * 0.7071, r * 0.7071), (h / 2 + r, 0.0)]
    a = np.arange(n) * 2 * np.pi / n
    v = []
    for z, rr in rings:
        for t in a:
            v.append((rr * math.cos(t), rr * math.sin(t), z))
    v = np.array(v)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a0, a1 = i * n + j, i * n + (j + 1) % n
            faces.append([a0, a1, a1 + n, a0 + n])
    return v, faces


_AXPERM = {"X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),  # local z -> x   (row vectors)
           "Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)}  # local z -> y


def _gather(stage: Usd.Stage, hide_fn, want_ground_only=False) -> tuple[_Polys, list]:
    """Traverse once, collect world polygons.  Returns (polys, camera-prim-list unused)."""
    out = _Polys()
    cols = _Colors(stage)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    batches: dict[tuple, list] = {}  # key -> list of (M(4,4), ci, flip)
    root = stage.GetPseudoRoot()
    it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in it:
        if prim == root:
            continue
        img = UsdGeom.Imageable(prim)
        if not img:
            continue
        path = prim.GetPath().pathString
        pa = img.GetPurposeAttr()
        if pa and pa.HasAuthoredValue() and pa.Get() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            it.PruneChildren()
            continue
        va = img.GetVisibilityAttr()
        if va and va.HasAuthoredValue() and va.Get() == UsdGeom.Tokens.invisible:
            it.PruneChildren()
            continue
        if hide_fn(prim, path):
            it.PruneChildren()
            continue
        t = prim.GetTypeName()
        if t not in ("Cube", "Cylinder", "Cone", "Capsule", "Sphere", "Mesh"):
            continue
        out.n_prims += 1
        M = np.array(xc.GetLocalToWorldTransform(prim), float)
        rgb, op = cols.get(prim)
        ci = out.cidx(rgb, op)
        flip = -1.0 if np.linalg.det(M[:3, :3]) < 0 else 1.0
        if t == "Cube":
            s = UsdGeom.Cube(prim).GetSizeAttr().Get() or 1.0
            batches.setdefault(("cube",), []).append((np.diag([s, s, s, 1.0]) @ M, ci, flip))
        elif t in ("Cylinder", "Cone"):
            g = UsdGeom.Cylinder(prim) if t == "Cylinder" else UsdGeom.Cone(prim)
            r, h = float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get())
            ax = g.GetAxisAttr().Get() or "Z"
            S = np.eye(4)
            S[:3, :3] = np.diag([r, r, h])
            if ax != "Z":
                S[:3, :3] = S[:3, :3] @ _AXPERM[ax]
            n = 8 if r * np.linalg.norm(M[:3, :3], axis=1).max() < 0.06 else 12
            batches.setdefault((t.lower(), n), []).append((S @ M, ci, flip))
        elif t == "Sphere":
            r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            batches.setdefault(("sphere",), []).append((np.diag([r, r, r, 1.0]) @ M, ci, flip))
        elif t == "Capsule":
            g = UsdGeom.Capsule(prim)
            r, h = float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get())
            ax = g.GetAxisAttr().Get() or "Z"
            v, faces = _capsule_polys(r, h, ax)
            if ax != "Z":
                v = v @ _AXPERM[ax]
            out.add(_world(v[_pad_faces(faces, 4)], M), ci, flip, False)
        else:
            _mesh_polys(prim, M, ci, out)
    for key, items in batches.items():
        Ms = np.array([m for m, _, _ in items])
        cis = np.array([c for _, c, _ in items])
        fl = np.array([f for _, _, f in items])
        if key[0] == "cube":
            v, f = _CUBE
            K, faces = 4, f
        elif key[0] == "sphere":
            v, f = _ICO
            K, faces = 4, _pad_faces(f, 4)
        else:
            v, f = _unit_prism(key[1], apex=key[0] == "cone")
            quads = [x for x in f if len(x) == 4]
            caps = [x for x in f if len(x) != 4]
            for K, faces in ((4, _pad_faces(quads, 4)), (12, _pad_faces(caps, 12))):
                _add_batch(out, v, faces, Ms, cis, fl, K)
            continue
        _add_batch(out, v, faces, Ms, cis, fl, K)
    return out, []


def _add_batch(out, v, faces, Ms, cis, fl, K):
    n = len(Ms)
    W = np.einsum("vk,nkj->nvj", np.c_[v, np.ones(len(v))], Ms)[..., :3]  # (n, nv, 3)
    V = W[:, faces]  # (n, nf, K, 3)
    nf = faces.shape[0]
    out.add_batch(V.reshape(n * nf, K, 3), np.repeat(cis, nf), np.repeat(fl, nf), np.zeros(n * nf, bool))


# ============================================================================ camera
@dataclass
class Cam:
    pos: np.ndarray
    right: np.ndarray
    up: np.ndarray
    back: np.ndarray
    focal: float = 18.0
    hap: float = 20.955
    near: float = 0.05
    far: float = 1e4
    ortho: bool = False
    ortho_width: float = 10.0  # world width covered by image (ortho)
    center_off: tuple = (0.0, 0.0)  # unused
    hide: list = field(default_factory=list)
    name: str = "camera"

    @property
    def B(self):
        return np.stack([self.right, self.up, self.back], 1)

    @property
    def forward(self):
        return -self.back


def _look_at(pos, target, up=(0, 0, 1)):
    pos, target, up = (np.asarray(x, float) for x in (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    if abs(float(fwd @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    u = np.cross(right, fwd)
    return pos, right, u, -fwd


def _env_ranges(graph: dict, stage=None):
    for k in ("environment", "building"):
        e = graph.get(k)
        if e and "x_range" in e and "y_range" in e:
            return list(e["x_range"]), list(e["y_range"])
    if stage is not None:
        from ..colliders import collect
        cs = collect(stage)
        cs = [c for c in cs if c.category not in ("roof",)]
        if cs:
            lo = np.min([c.obb.c - np.abs(c.obb.R * c.obb.h).sum(1) for c in cs], 0)
            hi = np.max([c.obb.c + np.abs(c.obb.R * c.obb.h).sum(1) for c in cs], 0)
            return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]
    return [-10.0, 10.0], [-10.0, 10.0]


def _resolve_camera(camera, graph: dict, stage: Usd.Stage, W: int, H: int) -> Cam:
    entry = None
    if isinstance(camera, str):
        for c in graph.get("cameras", []) or []:
            if camera in (c.get("name"), c.get("id"), c.get("path")):
                entry = c
                break
        if camera.startswith("/"):
            path = camera
        elif entry is not None and entry.get("path"):
            path = entry["path"]
        else:
            path = None
            for cand in (f"/World/Cameras/{camera}", f"/World/Cameras/{camera.capitalize()}"):
                if stage.GetPrimAtPath(cand):
                    path = cand
                    break
        if path is None or not stage.GetPrimAtPath(path):
            names = [c.get("name") or c.get("id") for c in graph.get("cameras", []) or []]
            raise ValueError(f"unknown camera {camera!r}; available: {names}")
        prim = stage.GetPrimAtPath(path)
        cam = UsdGeom.Camera(prim)
        if not cam:
            raise ValueError(f"{path} is not a UsdGeom.Camera")
        M = np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim), float)
        ax = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=1, keepdims=True)
        cr = cam.GetClippingRangeAttr().Get() or Gf.Vec2f(0.05, 1e4)
        hap = float(cam.GetHorizontalApertureAttr().Get() or 20.955)
        c = Cam(M[3, :3].copy(), ax[0], ax[1], ax[2], float(cam.GetFocalLengthAttr().Get() or 18.0), hap,
                max(float(cr[0]), 0.02), float(cr[1]), name=(entry or {}).get("name") or Path(path).name)
        if cam.GetProjectionAttr().Get() == UsdGeom.Tokens.orthographic:
            c.ortho, c.ortho_width = True, hap * 0.1
        if entry and entry.get("hide"):
            c.hide = list(entry["hide"])
        return c
    d = dict(camera)
    xr, yr = d.get("x_range"), d.get("y_range")
    if d.get("mode") == "top_down":
        if not (xr and yr):
            xr, yr = _env_ranges(graph, stage)
        cx, cy = (xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2
        wx, wy = (xr[1] - xr[0]) * 1.06, (yr[1] - yr[0]) * 1.06
        width = max(wx, wy * W / H)
        return Cam(np.array([cx, cy, 1000.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                   near=0.0, far=1e5, ortho=True, ortho_width=width, name=d.get("name", "topdown"), hide=list(d.get("hide", [])))
    pos, right, up, back = _look_at(d["position"], d["look_at"], d.get("up", (0, 0, 1)))
    ap = d.get("aperture_mm", [20.955, 11.79])
    return Cam(pos, right, up, back, float(d.get("focal_mm", 18.0)), float(ap[0]), float(d.get("near", 0.05)),
               float(d.get("far", 1e4)), name=d.get("name", "custom"), hide=list(d.get("hide", [])))


# ============================================================================ rasteriser
def _project(cam: Cam, P, W, H):
    """World points (...,3) -> (u, v, depth).  Pixel coords at size (W,H)."""
    v = (P - cam.pos) @ cam.B
    d = -v[..., 2]
    if cam.ortho:
        s = W / cam.ortho_width
        return W / 2 + s * v[..., 0], H / 2 - s * v[..., 1], d
    f = cam.focal / cam.hap * W
    dd = np.where(d > 1e-9, d, 1e-9)
    return W / 2 + f * v[..., 0] / dd, H / 2 - f * v[..., 1] / dd, d


def _newell(V):
    nxt = np.roll(V, -1, axis=1)
    return np.cross(V, nxt).sum(1)


def _clip_near(poly_view, near):
    """Sutherland-Hodgman on one polygon (n,3 view-space) against d=-z > near.  Returns list of points."""
    out = []
    n = len(poly_view)
    for i in range(n):
        a, b = poly_view[i], poly_view[(i + 1) % n]
        da, db = -a[2] - near, -b[2] - near
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def _rasterise(polys: _Polys, cam: Cam, W: int, H: int, bg=(238, 240, 243)):
    """Draw polys at supersampled size and return an RGB PIL image at (W,H)."""
    SW, SH = W * SSAA, H * SSAA
    img = Image.new("RGB", (SW, SH), bg)
    colors = np.array([c for c, _ in polys.colors] or [GREY], float)
    alphas = np.array([o for _, o in polys.colors] or [1.0], float)
    items = []  # (key, xy list, fill tuple, outline tuple|None, alpha)
    for K in (4, 12):
        g = polys.cat(K)
        if g is None:
            continue
        V, ci, flip, two = g
        P = len(V)
        nrm = _newell(V) * flip[:, None]
        ln = np.linalg.norm(nrm, axis=1)
        good = ln > 1e-12
        nrm = nrm / np.where(good, ln, 1)[:, None]
        cen = V.mean(1)
        if cam.ortho:
            tocam = np.broadcast_to(cam.back, (P, 3))
        else:
            tocam = cam.pos - cen
            tocam = tocam / np.maximum(np.linalg.norm(tocam, axis=1, keepdims=True), 1e-9)
        facing = (nrm * tocam).sum(1)
        sgn = np.where(two & (facing < 0), -1.0, 1.0)
        nrm = nrm * sgn[:, None]
        facing = facing * sgn
        keep = good & (facing > 1e-6)
        u, v, d = _project(cam, V, SW, SH)
        inz = d > cam.near if not cam.ortho else np.ones_like(d, bool)
        allin = inz.all(1)
        some = inz.any(1) & ~allin if not cam.ortho else np.zeros(P, bool)
        farok = (d < cam.far).any(1)
        keep &= farok
        # screen bbox culling (fully-in-front polygons)
        bx0, bx1, by0, by1 = u.min(1), u.max(1), v.min(1), v.max(1)
        onscr = (bx1 >= 0) & (bx0 <= SW) & (by1 >= 0) & (by0 <= SH) & ((bx1 - bx0 > 0.4) | (by1 - by0 > 0.4))
        sel = keep & allin & onscr
        # shading
        lam = np.clip(nrm @ KEY_LIGHT, 0, 1)
        shade = 0.36 + 0.52 * lam + 0.22 * np.clip(facing, 0, 1)
        zmax = V[..., 2].max(1)
        ground = zmax < 0.03
        dkey = 0.4 * d.mean(1) + 0.6 * d.max(1)
        key = np.where(ground, 1e9 - zmax * 1e5, dkey)
        rgbs = np.clip(colors[ci] * shade[:, None], 0, 1) * 255
        idx = np.nonzero(sel)[0]
        xy = np.stack([u[idx], v[idx]], -1).reshape(len(idx), -1).tolist() if len(idx) else []
        fill = rgbs[idx].astype(np.uint8).tolist()
        ci_i = ci[idx]
        for j, i in enumerate(idx):
            items.append((key[i], xy[j], fill[j], alphas[ci_i[j]]))
        # near-plane straddlers
        for i in np.nonzero(keep & some )[0]:
            vv = (V[i] - cam.pos) @ cam.B
            pts = _clip_near(vv, cam.near)
            if len(pts) < 3:
                continue
            pts = np.array(pts)
            f = cam.focal / cam.hap * SW
            dd = -pts[:, 2]
            xy_i = np.stack([SW / 2 + f * pts[:, 0] / dd, SH / 2 - f * pts[:, 1] / dd], -1)
            items.append((key[i], xy_i.reshape(-1).tolist(), rgbs[i].astype(np.uint8).tolist(), alphas[ci[i]]))
    items.sort(key=lambda t: -t[0])
    dr = ImageDraw.Draw(img)
    over = ImageDraw.Draw(img, "RGBA")
    for _, xy, fill, a in items:
        ol = (int(fill[0] * 0.55), int(fill[1] * 0.55), int(fill[2] * 0.55))
        if a >= 0.999:
            dr.polygon(xy, fill=tuple(fill), outline=ol)
        else:
            over.polygon(xy, fill=(*fill, int(255 * max(a, 0.15))), outline=(*ol, 200))
    return img.resize((W, H), Image.LANCZOS)


# ============================================================================ overlays
def _clip_seg(cam: Cam, a, b):
    """Clip a 3D segment to the near plane (perspective).  Returns (a,b) or None."""
    if cam.ortho:
        return a, b
    da, db = -((a - cam.pos) @ cam.B)[2] - cam.near, -((b - cam.pos) @ cam.B)[2] - cam.near
    if da < 0 and db < 0:
        return None
    if da < 0:
        a = a + (b - a) * (da / (da - db))
    elif db < 0:
        b = b + (a - b) * (db / (db - da))
    return a, b


def _annotate(img: Image.Image, cam: Cam, graph: dict):
    W, H = img.size
    dr = ImageDraw.Draw(img, "RGBA")
    font = _font(max(11, H // 45))

    def px(p):
        u, v, d = _project(cam, np.asarray(p, float), W, H)
        return float(u), float(v), float(d)

    def vis(p):
        u, v, d = px(p)
        return (cam.ortho or d > cam.near) and -50 <= u <= W + 50 and -50 <= v <= H + 50

    def label(x, y, text, fg, bg=(255, 255, 255, 190)):
        l, t, r, b = dr.textbbox((x, y), text, font=font)
        dr.rectangle((l - 2, t - 1, r + 2, b + 1), fill=bg)
        dr.text((x, y), text, fill=fg, font=font)

    # hazards: dashed polylines
    for h in graph.get("hazards", []) or []:
        wp = h.get("waypoints") or []
        if len(wp) < 2:
            continue
        z = float(h.get("z", 0.05))
        pts = [np.array([w[0], w[1], z]) for w in wp]
        if h.get("path_mode") == "loop":
            pts.append(pts[0])
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            n = max(1, int(L / 0.6))
            for k in range(0, n, 2):
                s, e = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
                c = _clip_seg(cam, s, e)
                if c is None:
                    continue
                (u0, v0, _), (u1, v1, _) = px(c[0]), px(c[1])
                dr.line((u0, v0, u1, v1), fill=(230, 40, 150, 230), width=3)
        if vis(pts[0]):
            u, v, _ = px(pts[0])
            label(u + 4, v + 4, str(h.get("kind") or Path(str(h.get("path", "hazard"))).name), (160, 0, 100, 255))
    # goals
    for g in graph.get("goals", []) or []:
        bp = g.get("base_pose") or g.get("position") or g.get("pose") or g.get("xy")
        if not bp or not vis([bp[0], bp[1], 0.05]):
            continue
        u, v, _ = px([bp[0], bp[1], 0.05])
        r = 5
        dr.polygon([(u, v - r - 2), (u + r + 2, v), (u, v + r + 2), (u - r - 2, v)], fill=(30, 110, 255, 235), outline=(255, 255, 255, 255))
        label(u + 7, v - 7, str(g.get("name", g.get("id", "goal"))), (10, 60, 180, 255))
    # spawn
    sp = (graph.get("robot") or {}).get("spawn")
    if sp and vis([sp[0], sp[1], 0.05]):
        u, v, _ = px([sp[0], sp[1], 0.05])
        dr.ellipse((u - 8, v - 8, u + 8, v + 8), fill=(30, 190, 70, 235), outline=(255, 255, 255, 255), width=2)
        if len(sp) >= 3:
            yaw = float(sp[2])
            e = [sp[0] + math.cos(yaw) * 0.9, sp[1] + math.sin(yaw) * 0.9, 0.05]
            u2, v2, _ = px(e)
            dr.line((u, v, u2, v2), fill=(255, 255, 255, 255), width=3)
        label(u + 10, v - 18, "robot spawn", (10, 110, 40, 255))
    return img


# ============================================================================ public API
def _default_out(scene_dir: Path, cam_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(cam_name)).strip("_") or "camera"
    return scene_dir / "renders" / f"proxy_{safe}.png"


def render_proxy(scene, camera, out_png=None, width: int = 960, height: int = 540, hide: list[str] | None = None,
                 annotate: bool = True) -> Path:
    """Render a flat-shaded preview of ``scene`` (scene dir or .usd) from ``camera`` (name | prim path | dict).

    Roof prims (paths containing "/Roof") are always hidden for overhead views (``{"mode": "top_down"}`` or a camera
    looking steeply downward).  ``hide``: path prefixes ("/World/X") or names/categories; defaults to the camera's
    scene-graph ``hide`` list.  Returns the PNG path."""
    scene_dir, graph, usd = _load_scene(scene)
    stage = Usd.Stage.Open(str(usd))
    cam = _resolve_camera(camera, graph, stage, width, height)
    overhead = cam.ortho or float(cam.forward[2]) < -0.8
    hides = list(hide) if hide is not None else list(cam.hide)
    polys, _ = _gather(stage, _make_hide(hides, overhead))
    img = _rasterise(polys, cam, width, height)
    if annotate:
        img = _annotate(img, cam, graph)
    out = Path(out_png) if out_png else _default_out(scene_dir, cam.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=False, compress_level=3)
    return out


ZONE_COLORS = [(255, 170, 0), (0, 170, 255), (160, 90, 255), (0, 200, 120), (255, 80, 120), (150, 150, 0)]


def _zone_rects(zones):
    out = []
    if isinstance(zones, dict):
        for k, v in zones.items():
            if isinstance(v, (list, tuple)) and len(v) == 4 and not isinstance(v[0], (list, dict)):
                out.append((k, v))
            elif isinstance(v, dict) and "rect" in v:
                out.append((k, v["rect"]))
    elif isinstance(zones, list):
        for z in zones:
            if isinstance(z, dict) and "rect" in z:
                out.append((str(z.get("id", z.get("name", "zone"))), z["rect"]))
    return out


def render_topdown_map(scene, out_png=None, px_per_m: float = 20.0) -> Path:
    """Orthographic semantic map: collider footprints in the robot band by category, zones, agent paths, goals, spawn,
    1 m grid, scale bar and legend."""
    from scipy.spatial import ConvexHull

    from ..colliders import collect

    scene_dir, graph, usd = _load_scene(scene)
    stage = Usd.Stage.Open(str(usd))
    xr, yr = _env_ranges(graph, stage)
    m = 1.0
    x0, x1, y0, y1 = xr[0] - m, xr[1] + m, yr[0] - m, yr[1] + m
    ppm = float(px_per_m)
    MW, MH = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    legend_w = 190
    img = Image.new("RGB", (MW + legend_w, max(MH, 260)), (250, 250, 250))
    dr = ImageDraw.Draw(img, "RGBA")
    font = _font(12)

    def P(x, y):
        return (x - x0) * ppm, (y1 - y) * ppm

    dr.rectangle((0, 0, MW, MH), fill=(244, 244, 240))
    # environment rect
    dr.rectangle((*P(xr[0], yr[1]), *P(xr[1], yr[0])), fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    # grid
    for gx in range(math.ceil(x0), math.floor(x1) + 1):
        a = P(gx, y0)
        b = P(gx, y1)
        dr.line((a[0], a[1], b[0], b[1]), fill=(0, 0, 0, 40 if gx % 5 else 90), width=1)
        if gx % 5 == 0:
            dr.text((a[0] + 2, a[1] - 13), str(gx), fill=(80, 80, 80), font=font)
    for gy in range(math.ceil(y0), math.floor(y1) + 1):
        a = P(x0, gy)
        b = P(x1, gy)
        dr.line((a[0], a[1], b[0], b[1]), fill=(0, 0, 0, 40 if gy % 5 else 90), width=1)
        if gy % 5 == 0:
            dr.text((2, a[1] + 1), str(gy), fill=(80, 80, 80), font=font)
    # zones
    for i, (name, r) in enumerate(_zone_rects((graph.get("layout") or {}).get("zones"))):
        c = ZONE_COLORS[i % len(ZONE_COLORS)]
        ax, ay = P(r[0], r[3])
        bx, by = P(r[2], r[1])
        dr.rectangle((ax, ay, bx, by), fill=(*c, 38), outline=(*c, 255), width=2)
        dr.text((ax + 4, ay + 3), str(name), fill=(*[int(k * 0.6) for k in c], 255), font=font)
    # colliders
    used: dict[str, tuple] = {}
    for c in sorted(collect(stage), key=lambda c: c.obb.c[2]):
        ext = np.abs(c.obb.R * c.obb.h).sum(1)
        zlo, zhi = c.obb.c[2] - ext[2], c.obb.c[2] + ext[2]
        cat = c.category or "other"
        if cat in ("floor", "roof", "camera") or zhi < 0.05 or zlo > 1.9:
            continue
        rgb = tuple(int(255 * k) for k in _category_color(cat if c.category else ""))
        if not c.category:
            rgb = (150, 150, 150)
        used[cat] = rgb
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * c.obb.h
        xy = (corners @ c.obb.R.T + c.obb.c)[:, :2]
        if c.shape in ("Cylinder", "Sphere", "Capsule") and abs(c.obb.R[2, 2]) > 0.9:
            t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            e = np.c_[np.cos(t) * c.obb.h[0], np.sin(t) * c.obb.h[1]] @ c.obb.R[:2, :2].T + c.obb.c[:2]
            hull = e
        else:
            try:
                hull = xy[ConvexHull(xy).vertices]
            except Exception:
                hull = xy
        pts = [P(x, y) for x, y in hull]
        dr.polygon(pts, fill=(*rgb, 255), outline=tuple(int(k * 0.55) for k in rgb))
    # hazard / agent paths
    for h in (graph.get("hazards") or []) + (graph.get("agents") or []):
        wp = h.get("waypoints") or h.get("path") or []
        if not isinstance(wp, list) or len(wp) < 2 or not isinstance(wp[0], (list, tuple)):
            continue
        pts = [P(w[0], w[1]) for w in wp]
        if h.get("path_mode") == "loop":
            pts.append(pts[0])
        for a, b in zip(pts[:-1], pts[1:]):
            L = math.dist(a, b)
            n = max(1, int(L / 8))
            for k in range(0, n, 2):
                s = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                e = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
                dr.line((*s, *e), fill=(230, 40, 150, 255), width=2)
        dr.text((pts[0][0] + 4, pts[0][1] + 3), str(h.get("kind") or Path(str(h.get("path", "agent"))).name), fill=(160, 0, 100), font=font)
    # goals + spawn
    for g in graph.get("goals") or []:
        bp = g.get("base_pose") or g.get("position") or g.get("pose") or g.get("xy")
        if not bp:
            continue
        u, v = P(bp[0], bp[1])
        dr.polygon([(u, v - 7), (u + 7, v), (u, v + 7), (u - 7, v)], fill=(30, 110, 255, 255), outline=(255, 255, 255, 255))
        dr.text((u + 8, v - 6), str(g.get("name", g.get("id", ""))), fill=(10, 60, 180), font=font)
    sp = (graph.get("robot") or {}).get("spawn")
    if sp:
        u, v = P(sp[0], sp[1])
        dr.ellipse((u - 8, v - 8, u + 8, v + 8), fill=(30, 190, 70, 255), outline=(255, 255, 255, 255), width=2)
        if len(sp) >= 3:
            e = P(sp[0] + math.cos(sp[2]) * 1.0, sp[1] + math.sin(sp[2]) * 1.0)
            dr.line((u, v, *e), fill=(255, 255, 255, 255), width=3)
        dr.text((u + 10, v - 18), "robot spawn", fill=(10, 110, 40), font=font)
    # scale bar
    bx, by = 12, MH - 18
    dr.rectangle((bx - 4, by - 14, bx + 5 * ppm + 40, by + 8), fill=(255, 255, 255, 220))
    dr.line((bx, by, bx + 5 * ppm, by), fill=(0, 0, 0, 255), width=3)
    dr.text((bx + 5 * ppm + 6, by - 7), "5 m", fill=(0, 0, 0), font=font)
    # legend
    lx, ly = MW + 10, 10
    dr.text((lx, ly), "legend", fill=(0, 0, 0), font=font)
    ly += 20
    items = list(sorted(used.items())) + [("goal", (30, 110, 255)), ("spawn", (30, 190, 70)), ("hazard path", (230, 40, 150))]
    for name, rgb in items:
        dr.rectangle((lx, ly, lx + 14, ly + 12), fill=(*rgb, 255), outline=(60, 60, 60))
        dr.text((lx + 22, ly), name, fill=(0, 0, 0), font=font)
        ly += 18
    out = Path(out_png) if out_png else scene_dir / "renders" / "proxy_topdown_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
