"""Measure the Isaac Sim people data that ``people.py`` needs and write ``library/people/*.json``.

    python -m worldc.assets.people_bake [--root DIR_OR_URL]

* ``isaac_5.1_characters.json``: per character its height, skeleton prim and the rest pose of every joint.
* ``isaac_5.1_gaits.json``: the clips' ``biped_demo`` rig, the standing clips (idle, look around: whole clips, they are closed loops), and for each walk clip the stretch of it that is one clean gait cycle (straight,
  steady speed, first pose == last pose), with the speed it was captured at so the playback can be matched to the walking speed.

Reads the assets through the catalog cache (downloaded once); nothing from NVIDIA's assets is committed except these measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdSkel

from .catalog import DEFAULT_CACHE, _fetch, catalog_url
from .people import fetch_asset

LIB = Path(__file__).parent / "library" / "people"
CLIPS = [f"stand_walk_{i}{m}" for i in (1, 2, 3, 4, 5, 7) for m in ("", "_mirror")]
STANDING = ["stand_idle_loop", "LookAround"]
FPS = 30.0


def fetch(path: str, root: str | None = None) -> Path:
    if root is None:
        return fetch_asset(path)
    dst = DEFAULT_CACHE / "people" / ("bake_" + path.replace("/", "__"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    _fetch(catalog_url(path, root), dst)
    return dst


def rest_rows(skel: UsdSkel.Skeleton):
    rows = []
    for m in skel.GetRestTransformsAttr().Get():
        q = m.ExtractRotationQuat()
        t = m.ExtractTranslation()
        rows.append([round(float(v), 6) for v in (*t, q.GetReal(), *q.GetImaginary())])
    return [str(j) for j in skel.GetJointsAttr().Get()], rows


def measure_character(name: str, root: str | None):
    stage = Usd.Stage.Open(str(fetch(f"Isaac/People/Characters/{name}/{name}.usd", root)))
    skel = next(UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    joints, rest = rest_rows(skel)
    prim = str(skel.GetPath()).split("/", 2)[2]  # below the default prim: that is where the reference lands
    return {"skeleton": prim, "joints": joints, "rest": rest}


def measure_biped(root: str | None):
    stage = Usd.Stage.Open(str(fetch("Isaac/People/Characters/biped_demo/biped_demo_meters.usd", root)))
    joints, rest = rest_rows(UsdSkel.Skeleton(stage.GetPrimAtPath("/Biped_Demo/Root")))
    return {"joints": joints, "rest": rest}


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:  # q (n, 4) as w,x,y,z
    w, u = q[:, :1], q[:, 1:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def load_clip(name: str, root: str | None):
    stage = Usd.Stage.Open(str(fetch(f"Isaac/People/Animations/{name}.skelanim.usd", root)))
    anim = UsdSkel.Animation(stage.GetPrimAtPath("/Root"))
    joints = [str(j) for j in anim.GetJointsAttr().Get()]
    times = anim.GetRotationsAttr().GetTimeSamples()
    rot = np.array([[(q.GetReal(), *q.GetImaginary()) for q in anim.GetRotationsAttr().Get(t)] for t in times])
    trn = np.array([[tuple(v) for v in anim.GetTranslationsAttr().Get(t)] for t in times])
    return joints, rot, trn


def find_cycle(joints, rot, trn, periods=range(34, 48)):
    """Best (start, frames, speed_mps, closure_deg, drift_deg): the window whose first and last pose match, walking straight and steadily."""
    short = [j.rsplit("/", 1)[-1] for j in joints]
    body = [i for i, s in enumerate(short) if s.endswith(("UpLeg", "LoLeg", "Ankle", "UpArm", "LoArm")) or s in ("Pelvis", "Spine1", "Spine2", "Chest", "Head")]
    sel = rot[:, body]
    sel = sel * np.sign((sel * sel[:1]).sum(-1, keepdims=True) + 1e-9)  # one hemisphere
    root = trn[:, 0][:, [0, 2]]
    fwd = _rotate(rot[:, 0], np.array([0.0, 0.0, 1.0]))
    heading = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    step = np.r_[0, np.linalg.norm(np.diff(root, axis=0), axis=1)] * FPS
    best = None
    for P in periods:
        for s in range(0, len(rot) - P):
            drift = abs(math.degrees(heading[s + P] - heading[s]))
            if drift > 3.0 or np.ptp(math.degrees(1) * heading[s:s + P + 1]) > 6.0:
                continue
            seg = step[s + 1:s + P + 1]
            if seg.min() < 0.6 or seg.max() > 1.6:
                continue
            dot = np.abs((sel[s] * sel[s + P]).sum(-1)).clip(0, 1)
            closure = math.degrees(np.mean(2 * np.arccos(dot)))
            score = closure + 0.3 * drift + 4 * float(seg.std())
            if best is None or score < best[0]:
                path = float(np.linalg.norm(np.diff(root[s:s + P + 1], axis=0), axis=1).sum())
                best = (score, s, P, path / (P / FPS), closure, drift)
    if best is None:
        return None
    _, s, P, v, closure, drift = best
    return {"start": s, "frames": P, "speed_mps": round(v, 3), "closure_deg": round(closure, 2), "drift_deg": round(drift, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="Isaac asset root (URL or directory); default WORLDC_ISAAC_ASSET_ROOT / the 5.1 S3 root")
    a = ap.parse_args(argv)
    old = json.loads((LIB / "isaac_5.1_characters.json").read_text())
    chars = {n: {"height_m": old[n]["height_m"], **measure_character(n, a.root)} for n in old}
    (LIB / "isaac_5.1_characters.json").write_text(json.dumps(chars, indent=None, separators=(",", ":")).replace("},\"", "},\n\"") + "\n")
    gaits = {"rig": measure_biped(a.root), "clips": {}}
    for name in CLIPS:
        joints, rot, trn = load_clip(name, a.root)
        assert joints == gaits["rig"]["joints"], name
        cyc = find_cycle(joints, rot, trn)
        print(name, cyc)
        if cyc and cyc["closure_deg"] <= 2.0 and cyc["drift_deg"] <= 2.0:  # keep only clean loops
            gaits["clips"][name] = cyc
    gaits["poses"] = {}  # standing clips: closed loops, the whole clip
    for name in STANDING:
        joints, rot, trn = load_clip(name, a.root)
        assert joints == gaits["rig"]["joints"], name
        gaits["poses"][name] = {"start": 0, "frames": len(rot) - 1}
    (LIB / "isaac_5.1_gaits.json").write_text(json.dumps(gaits, separators=(",", ":")) + "\n")
    print(f"{len(chars)} characters, {len(gaits['clips'])} gait cycles")


if __name__ == "__main__":
    main()
