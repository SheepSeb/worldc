"""Proxy (software) renderer: projection correctness, hide, determinism, top-down roof handling, perf."""
import json
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, UsdGeom

from conftest import build_asset
from worldc.assets import Pose
from worldc.render import render_proxy, render_topdown_map

BG = (238, 240, 243)
RED_PATH = "/World/RedBox"


def _is_red(px):
    r, g, b = (int(x) for x in px[:3])
    return r > 120 and r > g + 60 and r > b + 60


def _red_mask(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return (a[..., 0] > 120) & (a[..., 0] > a[..., 1] + 60) & (a[..., 0] > a[..., 2] + 60)


def _add_camera(stage, path, pos, look_at, focal=18.0):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(11.79)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    pos, look_at = np.array(pos, float), np.array(look_at, float)
    back = pos - look_at
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    m = Gf.Matrix4d(1.0)
    m.SetRow3(0, Gf.Vec3d(*right))
    m.SetRow3(1, Gf.Vec3d(*up))
    m.SetRow3(2, Gf.Vec3d(*back))
    m.SetTranslateOnly(Gf.Vec3d(*pos))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(m)


@pytest.fixture
def scene_dir(scratch_ctx, tmp_path):
    k = scratch_ctx.kit
    build_asset(scratch_ctx, "building_shell", {"size": [10, 8]})
    build_asset(scratch_ctx, "rack_row", pose=Pose.xy_yaw(-3.6, 0.0))
    for i in range(3):
        build_asset(scratch_ctx, "cardboard_box", {"size": [0.4, 0.3, 0.3]}, Pose.xy_yaw(2.5 + 0.5 * i, -2.5, 0.2 * i, 0.15))
    red = k.material("RedMat", color=(0.9, 0.05, 0.05))
    k.cube(RED_PATH, (0.5, 0.5, 0.5), (0.0, 1.0, 1.0), vmat=red)  # size-1 cube + scale op, like the generators
    k.scope("/World/Cameras")
    _add_camera(k.stage, "/World/Cameras/Front", (0.0, -3.0, 1.0), (0.0, 1.0, 1.0))
    k.save()
    graph = {
        "usd": "scene.usd",
        "environment": {"kind": "warehouse", "x_range": [-5, 5], "y_range": [-4, 4]},
        "cameras": [{"name": "front", "path": "/World/Cameras/Front"}],
        "robot": {"spawn": [3.0, 2.0, 1.0]},
        "goals": [{"id": 0, "name": "goal_a", "type": "pick", "base_pose": [-1.0, -2.0, 0.0]}],
        "hazards": [{"path": "/World/H", "kind": "human", "waypoints": [[-4, 3], [4, 3]], "z": 0.8, "path_mode": "pingpong"}],
        "layout": {"zones": {"dock": [-5, -4, -2, -2]}},
    }
    (tmp_path / "scene_graph.json").write_text(json.dumps(graph))
    return tmp_path


def _check_png(p, w, h):
    im = Image.open(p)
    assert im.size == (w, h)
    a = np.asarray(im.convert("RGB")).astype(float)
    assert a.std() > 8
    return a


def test_camera_by_name_path_dict(scene_dir):
    p1 = render_proxy(scene_dir, "front", width=480, height=270)
    assert p1 == scene_dir / "renders" / "proxy_front.png"
    _check_png(p1, 480, 270)
    p2 = render_proxy(scene_dir, "/World/Cameras/Front", out_png=scene_dir / "b.png", width=480, height=270)
    _check_png(p2, 480, 270)
    p3 = render_proxy(scene_dir, {"position": [0, -3, 1], "look_at": [0, 1, 1], "focal_mm": 18},
                      out_png=scene_dir / "c.png", width=480, height=270)
    _check_png(p3, 480, 270)
    # by usd-file path (no scene dir) works as well
    p4 = render_proxy(scene_dir / "scene.usd", "front", out_png=scene_dir / "d.png", width=480, height=270)
    assert p4.exists()
    # USD camera and the equivalent dict camera see the same image
    d = np.abs(np.asarray(Image.open(p2).convert("RGB")).astype(int) - np.asarray(Image.open(p3).convert("RGB")).astype(int))
    assert d.mean() < 3


def test_unknown_camera(scene_dir):
    with pytest.raises(ValueError):
        render_proxy(scene_dir, "nope")


def test_deterministic(scene_dir):
    a = render_proxy(scene_dir, "front", out_png=scene_dir / "a.png", width=320, height=180).read_bytes()
    b = render_proxy(scene_dir, "front", out_png=scene_dir / "a2.png", width=320, height=180).read_bytes()
    assert a == b


def test_projection_red_box_at_centre(scene_dir):
    W, H = 640, 360
    p = render_proxy(scene_dir, "front", out_png=scene_dir / "r.png", width=W, height=H, annotate=False)
    m = _red_mask(p)
    assert m.sum() > 200
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - W / 2) < 6 and abs(ys.mean() - H / 2) < 6
    # expected size: 0.5 m wide at depth 3.75 m (front face), f_px = 18/20.955*W
    f = 18 / 20.955 * W
    w_exp = f * 0.5 / 3.75
    assert abs((xs.max() - xs.min() + 1) - w_exp) < 0.12 * w_exp + 2
    # off-centre: look to the right, the box must move left in the image (and by about f*tan(14 deg))
    p2 = render_proxy(scene_dir, {"position": [0, -3, 1], "look_at": [1, 1, 1]}, out_png=scene_dir / "r2.png",
                      width=W, height=H, annotate=False)
    ys2, xs2 = np.nonzero(_red_mask(p2))
    assert abs((W / 2 - xs2.mean()) - f * 0.25) < 0.15 * f * 0.25


def test_hide(scene_dir):
    p = render_proxy(scene_dir, "front", out_png=scene_dir / "h0.png", width=320, height=180, annotate=False)
    assert _red_mask(p).sum() > 50
    for h in ([RED_PATH], ["RedBox"]):
        p = render_proxy(scene_dir, "front", out_png=scene_dir / "h1.png", width=320, height=180, hide=h, annotate=False)
        assert _red_mask(p).sum() == 0


def test_topdown_hides_roof(scene_dir):
    W, H = 960, 540
    p = render_proxy(scene_dir, {"mode": "top_down"}, out_png=scene_dir / "t.png", width=W, height=H, hide=[], annotate=False)
    a = _check_png(p, W, H)
    scale = W / (max(10 * 1.06, 8 * 1.06 * W / H))
    u, v = int(W / 2 + 0 * scale), int(H / 2 - 1.0 * scale)  # red box at world (0, 1)
    assert _red_mask(p)[v - 3:v + 4, u - 3:u + 4].mean() > 0.8  # visible => roof hidden
    # floor (a mesh) is drawn: an empty floor spot differs from the background
    fu, fv = int(W / 2 + 3.5 * scale), int(H / 2 + 3.0 * scale)
    assert np.abs(a[fv, fu] - np.array(BG)).sum() > 15
    # hiding the whole building leaves background at that spot
    p2 = render_proxy(scene_dir, {"mode": "top_down"}, out_png=scene_dir / "t2.png", width=W, height=H,
                      hide=["/World/Test"], annotate=False)
    a2 = np.asarray(Image.open(p2).convert("RGB")).astype(float)
    assert np.abs(a2[fv, fu] - np.array(BG)).sum() < 6


def test_annotations_change_image(scene_dir):
    p1 = render_proxy(scene_dir, {"mode": "top_down"}, out_png=scene_dir / "n1.png", width=480, height=270, annotate=False)
    p2 = render_proxy(scene_dir, {"mode": "top_down"}, out_png=scene_dir / "n2.png", width=480, height=270, annotate=True)
    assert p1.read_bytes() != p2.read_bytes()
    a = np.asarray(Image.open(p2).convert("RGB")).astype(int)
    green = (a[..., 1] > 150) & (a[..., 0] < 80) & (a[..., 2] < 120)
    assert green.sum() > 20  # spawn marker


def test_topdown_map(scene_dir):
    t = time.time()
    p = render_topdown_map(scene_dir, px_per_m=20)
    assert time.time() - t < 5
    assert p == scene_dir / "renders" / "proxy_topdown_map.png"
    im = Image.open(p)
    assert im.size[0] >= 12 * 20 and im.size[1] >= 10 * 20
    assert np.asarray(im.convert("RGB")).std() > 10


def test_small_scene_is_fast(scene_dir):
    t = time.time()
    render_proxy(scene_dir, "front")
    assert time.time() - t < 3


@pytest.mark.slow
def test_warehouse_overhead_perf(tmp_path):
    subprocess.run([sys.executable, "-m", "worldc.generate", "--seed", "0", "--out", str(tmp_path)],
                   check=True, capture_output=True)
    d = tmp_path / "seed_0"
    t = time.time()
    p = render_proxy(d, "overhead")
    dt = time.time() - t
    print(f"warehouse overhead render: {dt:.2f}s")
    assert dt < 15
    a = _check_png(p, 960, 540)
    assert a.std() > 20
    t = time.time()
    render_proxy(d, "aisle_level")
    assert time.time() - t < 15
