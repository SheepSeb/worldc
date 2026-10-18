"""Lighting presets, generalised to any footprint (ported from the warehouse and plant generators).

bright_day / mixed_warm / dim_night: grid of ceiling RectLights (some randomly off) + fill dome  (indoor)
indoor_industrial: fewer, stronger fixtures + bluish ambient dome
sun / overcast: distant sun light + coloured sky dome (outdoor, or indoor with daylight through openings)
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdLux

LIGHT_BASE = 30000.0  # RectLight intensity of a full-power ceiling panel
PANEL_CFG = {
    "bright_day": dict(keep=1.0, inten=1.0, temp=5600, dome=450.0),
    "mixed_warm": dict(keep=0.8, inten=0.9, temp=3800, dome=250.0),
    "dim_night": dict(keep=0.55, inten=0.42, temp=3000, dome=130.0),
}


def _q_down_to(d):
    """Quaternion turning a light's -Z towards direction d."""
    from .assets.procedural._assembly import qalign

    return qalign(-np.asarray(d, float))


def _q_face(n):
    """Quaternion turning a RectLight's emission axis (-Z) towards the horizontal direction n."""
    from .assets.procedural._assembly import qalign

    return qalign(-np.asarray([n[0], n[1], 0.0], float))


def build_lights(kit, rng, preset: str, x_range, y_range, height: float | None, scale: float = 1.0, color_temperature=None,
                 sky_texture: str | None = None, sun_azimuth_deg: float | None = None, sun_elevation_deg: float | None = None,
                 windows: list | None = None) -> dict:
    kit.scope("/World/Lights")
    info = {"setup_name": preset, "scale": scale}
    if preset in PANEL_CFG:
        cfg = PANEL_CFG[preset]
        h = (height or 8.0) - 0.35
        xs = np.arange(x_range[0] + 3.0, x_range[1], 6.0)
        ny = max(2, int(round((y_range[1] - y_range[0]) / 5.0)))
        ys = np.linspace(y_range[0] + 2.5, y_range[1] - 1.5, ny)
        panels = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if rng.random() > cfg["keep"]:
                    continue
                inten = LIGHT_BASE * cfg["inten"] * float(rng.uniform(0.85, 1.1)) * scale
                lp = f"/World/Lights/Panel_{i}_{j}"
                L = UsdLux.RectLight.Define(kit.stage, lp)
                L.CreateWidthAttr(1.4)
                L.CreateHeightAttr(0.5)
                L.CreateIntensityAttr(inten)
                L.CreateEnableColorTemperatureAttr(True)
                L.CreateColorTemperatureAttr(float((color_temperature or cfg["temp"]) + rng.uniform(-300, 300)))
                kit.set_pose(L.GetPrim(), (float(x), float(y), h))  # RectLight emits along -Z
                panels.append({"path": lp, "position": [float(x), float(y), h], "intensity": inten})
        dome = UsdLux.DomeLight.Define(kit.stage, "/World/Lights/FillDome")
        dome.CreateIntensityAttr(cfg["dome"] * scale)
        dome.CreateColorTemperatureAttr(float(color_temperature or cfg["temp"]))
        dome.CreateEnableColorTemperatureAttr(True)
        info.update(n_panels=len(panels), dome_intensity=cfg["dome"] * scale, panels=panels)
    elif preset == "indoor_industrial":
        dome = UsdLux.DomeLight.Define(kit.stage, "/World/Lights/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(0.55, 0.6, 0.7))
        dome.CreateIntensityAttr(float(260.0 * scale))
        h = (height or 5.65) - 0.35
        n = 0
        for i, x in enumerate(np.arange(x_range[0] + 2.0, x_range[1], 4.0)):
            for j, y in enumerate(np.arange(y_range[0] + 2.0, y_range[1], 4.0)):
                L = UsdLux.RectLight.Define(kit.stage, f"/World/Lights/Fixture_{i}_{j}")
                L.CreateWidthAttr(1.2)
                L.CreateHeightAttr(0.3)
                L.CreateIntensityAttr(float(60000.0 * rng.uniform(0.8, 1.1) * scale))
                L.CreateColorTemperatureAttr(float(color_temperature or 4200.0))
                L.CreateEnableColorTemperatureAttr(True)
                kit.set_pose(L.GetPrim(), (float(x), float(y), h))
                n += 1
        info.update(dome_intensity=260.0 * scale, fixtures=n)
    else:  # sun / overcast
        clear = preset == "sun"
        dome = UsdLux.DomeLight.Define(kit.stage, "/World/Lights/Sky")
        if sky_texture:
            dome.CreateTextureFileAttr(Sdf.AssetPath(sky_texture))
            dome.CreateTextureFormatAttr("latlong")
        else:
            dome.CreateColorAttr(Gf.Vec3f(*((0.62, 0.74, 0.95) if clear else (0.78, 0.8, 0.84))))
        dome.CreateIntensityAttr(float((1800.0 if clear else 1900.0) * rng.uniform(0.9, 1.1) * scale))
        sun = UsdLux.DistantLight.Define(kit.stage, "/World/Lights/Sun")
        az = float(rng.uniform(-2.5, -1.9))
        el = float(rng.uniform(0.7, 0.95)) if clear else 1.2
        if sun_azimuth_deg is not None:
            az = math.radians(sun_azimuth_deg)
        if sun_elevation_deg is not None:
            el = math.radians(sun_elevation_deg)
        d = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        kit.set_pose(sun.GetPrim(), (0, 0, 10), _q_down_to(d))
        sun.CreateAngleAttr(1.0 if clear else 6.0)
        sun.CreateIntensityAttr(float((4200.0 if clear else 500.0) * scale))
        sun.CreateColorTemperatureAttr(float(color_temperature or (5900.0 if clear else 6800.0)))
        sun.CreateEnableColorTemperatureAttr(True)
        for i, (c, n, ww, wh) in enumerate(windows or []):  # skylight portal: soft area light in each window plane, facing in
            L = UsdLux.RectLight.Define(kit.stage, f"/World/Lights/Window_{i}")
            L.CreateWidthAttr(float(ww))
            L.CreateHeightAttr(float(wh))
            L.CreateIntensityAttr(float(9000.0 * scale))
            L.CreateColorTemperatureAttr(6500.0)
            L.CreateEnableColorTemperatureAttr(True)
            kit.set_pose(L.GetPrim(), tuple(c), _q_face(n))
        info.update(dome_intensity=dome.GetIntensityAttr().Get(), sun_intensity=sun.GetIntensityAttr().Get(),
                    sun_dir=[round(float(c), 3) for c in d])
    return info
