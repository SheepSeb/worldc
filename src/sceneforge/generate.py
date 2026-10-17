"""Procedural warehouse generator CLI (kept for compatibility; now runs the warehouse recipe through the spec compiler).

    uv run python -m sceneforge.generate --seed 0 1 2 --out warehouse_out [--lighting 0|1|2] [--light-scale 1.5]

Writes ``<out>/seed_N/{warehouse.usd, scene_graph.json, resolved_spec.json, textures/}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

VERSION = "2.0"


class GenerationError(RuntimeError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues if i.severity == "error"))


def build(seed: int, out_root: Path, lighting: int | None = None, light_scale: float = 1.0, run_check: bool = False) -> Path:
    from .compile import compile_scene
    from .recipes import warehouse

    out = Path(out_root) / f"seed_{seed}"
    res = compile_scene(warehouse(seed, lighting=lighting, light_scale=light_scale), seed, out, run_check=run_check)
    if not res.ok:
        raise GenerationError(res.issues)
    return out


def main(argv=None):
    from .isaac_env import gate_or_exit

    gate_or_exit()
    ap = argparse.ArgumentParser(description="Generate a procedural Isaac Sim warehouse (USD + scene graph).")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="warehouse_out")
    ap.add_argument("--lighting", type=int, choices=[0, 1, 2], default=None, help="0 bright, 1 mixed warm, 2 dim (default: seed %% 3)")
    ap.add_argument("--light-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    for s in a.seed:
        p = build(s, Path(a.out), a.lighting, a.light_scale)
        print(f"seed {s}: wrote {p}/warehouse.usd")


if __name__ == "__main__":
    main()
