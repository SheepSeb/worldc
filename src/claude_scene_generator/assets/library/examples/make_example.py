"""Regenerate the example library asset (a small open crate) with the importer:
    uv run python src/claude_scene_generator/assets/library/examples/make_example.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

from claude_scene_generator.assets.mesh.import_mesh import import_mesh


def crate_mesh(w=0.6, d=0.4, h=0.3, t=0.02) -> trimesh.Trimesh:
    """Open-top crate: floor plus four walls, built from boxes (non-convex)."""
    def box(size, c):
        b = trimesh.creation.box(extents=size)
        b.apply_translation(c)
        return b

    parts = [box((w, d, t), (0, 0, t / 2)),
             box((w, t, h), (0, d / 2 - t / 2, h / 2)), box((w, t, h), (0, -d / 2 + t / 2, h / 2)),
             box((t, d - 2 * t, h), (w / 2 - t / 2, 0, h / 2)), box((t, d - 2 * t, h), (-w / 2 + t / 2, 0, h / 2))]
    return trimesh.util.concatenate(parts)


def main():
    out = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "example_crate.stl"
        crate_mesh().export(src)
        sc = import_mesh(src, out, "example_crate", ["crate", "plastic", "container"], 3.0, material="plastic",
                         description="Small open-top plastic crate (0.6 x 0.4 x 0.3 m), dynamic, convex-hull compound collider.",
                         max_hulls=8)
    print(sc)


if __name__ == "__main__":
    main()
