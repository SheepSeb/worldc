"""Import a mesh file as a library asset: normalise (metres, Z-up, origin bottom-centre), decompose into convex hulls,
write visual USD + collision USD + sidecar YAML + footprint thumbnail.

    python -m worldc.assets.mesh.import_mesh in.glb --id cart_01 --tags cart,metal --mass 12
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from ...materials import PHYSICAL
from .mesh_asset import (DEFAULT_LIBRARY, AnchorSpec, Bounds, CollisionSpec, MeshSidecar, Original, Provenance,
                         write_sidecar)

IMPORTER_VERSION = "1"
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
USD_EXT = {".usd", ".usda", ".usdc", ".usdz"}
TRIMESH_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off"}
DEFAULT_DENSITY = 200.0  # kg/m3 of hull volume when no mass is given
MIN_TOP_AREA_FRAC = 0.25


class ImportError_(RuntimeError):
    pass


# --------------------------------------------------------------------------- loading
def _load_usd(path: Path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ImportError_(f"cannot open USD {path}")
    up = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    parts = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if root and root.IsValid() and not (prim.GetPath() == root.GetPath() or prim.GetPath().HasPrefix(root.GetPath())):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        counts, idx = m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        P = np.array(pts, float).reshape(-1, 3)
        M = np.array(cache.GetLocalToWorldTransform(prim))
        P = P @ M[:3, :3] + M[3, :3]
        tris, o = [], 0
        for c in counts:
            for j in range(1, c - 1):
                tris.append((idx[o], idx[o + j], idx[o + j + 1]))
            o += c
        F = np.array(tris, int)
        if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            F = F[:, ::-1]
        if np.linalg.det(M[:3, :3]) < 0:
            F = F[:, ::-1]
        color = None
        dc = m.GetDisplayColorAttr().Get()
        if dc and len(dc):
            color = tuple(float(c) for c in np.mean(np.array(dc, float), axis=0))
        parts.append((P, F, color))
    if not parts:
        raise ImportError_(f"no renderable meshes found in {path}")
    return parts, up, mpu


def _load_trimesh(path: Path):
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise ImportError_("importing glb/gltf/obj/stl needs the optional 'mesh' extra: uv sync --extra mesh") from e
    scene = trimesh.load(str(path), force="scene")
    parts = []
    for g in scene.dump(concatenate=False):
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        color = None
        try:
            vc = g.visual.to_color().vertex_colors
            color = tuple(float(c) / 255.0 for c in np.mean(vc[:, :3], axis=0))
        except Exception:
            pass
        parts.append((np.asarray(g.vertices, float), np.asarray(g.faces, int), color))
    if not parts:
        raise ImportError_(f"no triangle meshes found in {path}")
    up = "Y" if path.suffix.lower() in {".glb", ".gltf"} else "Z"
    return parts, up, 1.0


def load_parts(path: Path):
    ext = path.suffix.lower()
    if ext in USD_EXT:
        return _load_usd(path)
    if ext in TRIMESH_EXT:
        return _load_trimesh(path)
    raise ImportError_(f"unsupported mesh format {ext!r}")


# --------------------------------------------------------------------------- normalisation
def normalise(parts, up: str, mpu: float):
    """-> (parts in metres, Z-up, origin bottom-centre of the bounding footprint)."""
    out = []
    for P, F, c in parts:
        P = P * mpu
        if up.upper() == "Y":
            P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)  # rotate +90 deg about X
        out.append((P, F, c))
    allp = np.concatenate([p for p, _, _ in out])
    lo, hi = allp.min(0), allp.max(0)
    shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return [(P - shift, F, c) for P, F, c in out]


# --------------------------------------------------------------------------- collision
def _hull_3d(points: np.ndarray):
    from scipy.spatial import ConvexHull

    h = ConvexHull(points)
    V = points
    F = []
    for s, eq in zip(h.simplices, h.equations):
        a, b, c = V[s[0]], V[s[1]], V[s[2]]
        F.append(s if np.dot(np.cross(b - a, c - a), eq[:3]) > 0 else s[::-1])
    used = np.unique(np.array(F))
    remap = {int(v): i for i, v in enumerate(used)}
    return V[used], np.array([[remap[int(i)] for i in f] for f in F]), float(h.volume)


def decompose(V: np.ndarray, F: np.ndarray, max_hulls: int, mode: str = "auto", seed: int = 0):
    """-> (hulls [(verts, faces)], method, warnings)."""
    warnings = []
    if mode == "box":
        lo, hi = V.min(0), V.max(0)
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        hv, hf, _ = _hull_3d(c)
        return [(hv, hf)], "box", warnings
    if mode == "auto" and max_hulls > 1:
        try:
            import coacd

            coacd.set_log_level("error")
            res = coacd.run_coacd(coacd.Mesh(V, F.astype(np.int32)), threshold=0.05, max_convex_hull=max_hulls, seed=seed)
            hulls = [(np.asarray(v, float), np.asarray(f, int)) for v, f in res if len(f)]
            if hulls:
                return hulls, "coacd", warnings
            warnings.append("coacd returned no hulls; used a single convex hull")
        except ImportError:
            warnings.append("coacd not installed (extra 'mesh'); used a single convex hull")
        except Exception as e:  # coacd failure must not kill the import
            warnings.append(f"coacd failed ({type(e).__name__}: {e}); used a single convex hull")
    hv, hf, _ = _hull_3d(V)
    return [(hv, hf)], "single_hull", warnings


def _hull_volume(hv, hf) -> float:
    c = hv.mean(0)
    a, b, d = hv[hf[:, 0]] - c, hv[hf[:, 1]] - c, hv[hf[:, 2]] - c
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, d)).sum()) / 6.0)


def footprint_polygon(V: np.ndarray) -> list:
    from scipy.spatial import ConvexHull

    xy = V[:, :2]
    h = ConvexHull(xy)
    return [(round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)) for i in h.vertices]  # CCW in 2-D


def detect_top_anchor(parts, fp) -> AnchorSpec | None:
    """A large flat upward-facing face cluster (highest one with area >= 25 % of the footprint) -> support anchor."""
    from scipy.spatial import ConvexHull

    fp_pts = np.array(fp)
    fp_area = float(ConvexHull(fp_pts).volume)
    height = max(float(P[:, 2].max()) for P, _, _ in parts)
    levels: dict[int, list] = {}
    for P, F, _ in parts:
        a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n = np.cross(b - a, c - a)
        ar = np.linalg.norm(n, axis=1) / 2
        ok = (ar > 1e-10) & (n[:, 2] / np.maximum(2 * ar, 1e-12) > 0.98)
        for i in np.where(ok)[0]:
            z = (a[i, 2] + b[i, 2] + c[i, 2]) / 3
            levels.setdefault(int(round(z / 0.01)), []).append((ar[i], np.stack([a[i], b[i], c[i]])))
    for key in sorted(levels, reverse=True):
        tris = levels[key]
        z_lvl = key * 0.01
        if z_lvl < 0.8 * height:
            return None  # best flat face is deep inside (e.g. an open crate's floor), not a top
        if sum(t[0] for t in tris) >= MIN_TOP_AREA_FRAC * fp_area:
            pts = np.concatenate([t[1] for t in tris])
            z = float(np.mean([t[1][:, 2].mean() for t in tris]))
            xy = pts[:, :2]
            c = (xy.min(0) + xy.max(0)) / 2
            try:
                h = xy[ConvexHull(xy).vertices]
            except Exception:
                return None
            return AnchorSpec(name="top", kind="support", pos=(round(float(c[0]), 4), round(float(c[1]), 4), round(z, 4)),
                              polygon=[(round(float(x - c[0]), 4), round(float(y - c[1]), 4)) for x, y in h])
    return None


# --------------------------------------------------------------------------- writers
def _write_mesh_usd(path: Path, root: str, meshes: list, with_color: bool):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    r = UsdGeom.Xform.Define(stage, f"/{root}")
    stage.SetDefaultPrim(r.GetPrim())
    for i, (P, F, c) in enumerate(meshes):
        m = UsdGeom.Mesh.Define(stage, f"/{root}/{'m' if with_color else 'hull'}_{i:02d}")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in P]))
        m.CreateFaceVertexCountsAttr([3] * len(F))
        m.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(F).ravel()])
        m.CreateExtentAttr([Gf.Vec3f(*map(float, P.min(0))), Gf.Vec3f(*map(float, P.max(0)))])
        m.CreateSubdivisionSchemeAttr("none")
        if with_color:
            col = c if c is not None else (0.7, 0.7, 0.72)
            m.CreateDisplayColorAttr([Gf.Vec3f(*col)])
    stage.GetRootLayer().Save()


def _thumbnail(path: Path, fp, hulls, size: int = 256):
    from PIL import Image, ImageDraw

    pts = np.array(fp)
    lo, hi = pts.min(0), pts.max(0)
    s = (size - 24) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    ctr = (lo + hi) / 2

    def px(p):
        return (size / 2 + (p[0] - ctr[0]) * s, size / 2 - (p[1] - ctr[1]) * s)

    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.polygon([px(p) for p in fp], fill=(205, 215, 230), outline=(40, 60, 100))
    from scipy.spatial import ConvexHull

    for hv, _ in hulls:
        xy = hv[:, :2]
        try:
            poly = [px(xy[i]) for i in ConvexHull(xy).vertices]
            d.polygon(poly, outline=(200, 80, 40))
        except Exception:
            pass
    img.save(path)


# --------------------------------------------------------------------------- library function
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_mesh(src, out_dir, asset_id: str | None = None, tags=(), mass_kg: float | None = None, *,
                description: str | None = None, material: str = "steel", body: str = "dynamic", max_hulls: int = 16,
                up: str | None = None, units: str | None = None, friction: float | None = None,
                collision: str = "auto", seed: int = 0, thumbnail: bool = True) -> Path:
    """Import ``src`` into ``out_dir``; returns the sidecar path.  ``collision``: auto (coacd, else single hull) |
    single_hull | box."""
    src, out_dir = Path(src), Path(out_dir)
    if not src.is_file():
        raise ImportError_(f"no such file: {src}")
    if material not in PHYSICAL:
        raise ImportError_(f"unknown material {material!r}; choose from {sorted(PHYSICAL)}")
    if collision not in ("auto", "single_hull", "box"):
        raise ImportError_(f"collision must be auto|single_hull|box, got {collision!r}")
    asset_id = asset_id or re.sub(r"\W+", "_", src.stem).strip("_").lower()
    if not re.fullmatch(r"[A-Za-z_]\w*", asset_id):
        raise ImportError_(f"asset id {asset_id!r} must be an identifier ([A-Za-z_][A-Za-z0-9_]*)")
    out_dir.mkdir(parents=True, exist_ok=True)

    parts, src_up, src_mpu = load_parts(src)
    if up:
        src_up = up.upper()
    if units:
        if units not in UNITS:
            raise ImportError_(f"units must be one of {sorted(UNITS)}")
        src_mpu = UNITS[units]
    parts = normalise(parts, src_up, src_mpu)
    V = np.concatenate([p for p, _, _ in parts])
    F = np.concatenate([f + sum(len(p) for p, _, _ in parts[:i]) for i, (_, f, _) in enumerate(parts)])

    hulls, method, warnings = decompose(V, F, max_hulls, collision, seed)
    fp = footprint_polygon(V)
    lo, hi = V.min(0), V.max(0)
    if mass_kg is None:
        vol = sum(_hull_volume(hv, hf) for hv, hf in hulls)
        mass_kg = round(float(np.clip(vol * DEFAULT_DENSITY, 0.1, 5000.0)), 3)
        warnings.append(f"mass not given; estimated {mass_kg} kg from hull volume at {DEFAULT_DENSITY:g} kg/m3")

    vis_name, col_name = f"{asset_id}.usd", f"{asset_id}.collision.usd"
    _write_mesh_usd(out_dir / vis_name, "Visual", parts, True)
    _write_mesh_usd(out_dir / col_name, "Collision", [(hv, hf, None) for hv, hf in hulls], False)
    thumb = None
    if thumbnail:
        thumb = f"{asset_id}.png"
        _thumbnail(out_dir / thumb, fp, hulls)
    top = detect_top_anchor(parts, fp)
    ctype = "box" if method == "box" else ("convex_hulls" if len(hulls) > 1 else "convex_hull")
    sc = MeshSidecar(
        id=asset_id, description=description or f"Imported mesh '{src.stem}' ({', '.join(tags) or 'untagged'}).",
        tags=list(tags), source=vis_name, original=Original(up_axis=src_up, meters_per_unit=src_mpu),
        bounds=Bounds(min=tuple(round(float(v), 4) for v in lo), max=tuple(round(float(v), 4) for v in hi)),
        footprint=fp, mass_kg=mass_kg, material=material, friction=friction, body=body,
        collision=CollisionSpec(type=ctype, file=col_name, n_hulls=len(hulls)),
        anchors=[top] if top else [], thumbnail=thumb,
        provenance=Provenance(original_file=src.name, sha256=sha256_file(src), importer_version=IMPORTER_VERSION,
                              decomposition=method, warnings=warnings))
    return write_sidecar(sc, out_dir / f"{asset_id}.asset.yaml")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--id")
    ap.add_argument("--tags", default="", help="comma-separated")
    ap.add_argument("--mass", type=float, help="kg (default: estimated from hull volume)")
    ap.add_argument("--out", type=Path, default=DEFAULT_LIBRARY / "imported")
    ap.add_argument("--material", default="steel", choices=sorted(PHYSICAL))
    ap.add_argument("--body", default="dynamic", choices=["static", "dynamic", "kinematic"])
    ap.add_argument("--max-hulls", type=int, default=16)
    ap.add_argument("--up", choices=["y", "z"])
    ap.add_argument("--units", choices=sorted(UNITS))
    ap.add_argument("--collision", default="auto", choices=["auto", "single_hull", "box"])
    ap.add_argument("--description")
    a = ap.parse_args(argv)
    from ...isaac_env import gate_or_exit

    gate_or_exit()
    try:
        p = import_mesh(a.src, a.out, a.id, [t for t in a.tags.split(",") if t], a.mass, description=a.description,
                        material=a.material, body=a.body, max_hulls=a.max_hulls, up=a.up, units=a.units, collision=a.collision)
    except ImportError_ as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(p)


if __name__ == "__main__":
    main()
