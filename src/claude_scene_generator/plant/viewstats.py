"""Scene-density statistics of a rendered view from an instance-id segmentation image (prim path per pixel)."""
from __future__ import annotations

import re

import numpy as np

MIN_PX = 30


def classify(path: str):
    """(object key, category) for a leaf prim path, or (None, 'empty') for floor/ground/background."""
    if path in ("BACKGROUND", "UNLABELLED", ""):
        return None, "empty"
    p = path.split("/")
    # p[0] == '' ; p[1] == 'World'
    if len(p) < 4:
        return path, "other"
    sec = p[2]
    if sec == "Ground":
        return (None, "empty") if p[3] in ("visual", "collision") else (f"/World/Ground/{p[3]}", "background_plant")
    if sec == "Deck":
        if p[3].startswith(("grating_visual", "underside", "sump", "collision")):
            return None, "empty"
        return f"/World/Deck/{p[3]}", "deck_structure"
    if sec == "Background":
        return f"/World/Background/{p[3]}", "background_plant"
    if sec == "Structure":
        return f"/World/Structure/{p[3]}", "steel_structure"
    if sec == "Equipment":
        n = p[3]
        if n.startswith("GateValve"):
            return f"/World/Equipment/{n}", "valve_handwheel" if (len(p) > 4 and p[4] in ("wheel",)) else "valve_body"
        if n.startswith("ControlValve"):
            return f"/World/Equipment/{n}", "control_valve"
        if n.startswith(("PumpSet", "PumpManifold")):
            return f"/World/Equipment/{n}", "machine"
        if n.startswith(("Panel_", "BlueCabinet", "StainlessBox")):
            return f"/World/Equipment/{n}", "enclosure"
        return f"/World/Equipment/{n}", "equipment_other"
    if sec == "Pipes":
        n = p[4] if len(p) > 4 else p[3]
        if p[3] == "net":
            if n.startswith("fit"):
                return f"/World/Pipes/net/{n}", "pipe_fitting"
            if re.search(r"_s\d+$", n) or n.startswith(("rack0_post", "rack1_post")) or "_post" in n or "_tag" in n:
                return f"/World/Pipes/net/{n}", "pipe_support"
            return f"/World/Pipes/net/{n}", "pipe_segment"
        return f"/World/Pipes/{p[3]}/{n}", "pipe_segment"
    if sec == "Instruments":
        n = p[3]
        if n.startswith("Gauge"):
            return f"/World/Instruments/{n}", "gauge"
        if n.startswith("InstBox"):
            return f"/World/Instruments/{n}", "instrument_box"
        if n.startswith("TubingBundle") or n == "tubing":
            return f"/World/Instruments/{n}", "tubing_bundle"
        return f"/World/Instruments/{n}", "instrument_other"
    if sec == "Panel":
        n = p[3]
        if n.startswith("Cable_"):
            return f"/World/Panel/{n}", "cable_bundle"
        return f"/World/Panel/{n}", "panel_structure"
    if sec == "Worker":
        return "/World/Worker", "human"
    if sec == "Hoses":
        return f"/World/Hoses/{p[3]}", "hose"
    if sec == "Loose":
        return f"/World/Loose/{p[3]}", "loose_item"
    return path, "other"


def analyze(seg: np.ndarray, id_to_label: dict, min_px: int = MIN_PX) -> dict:
    ids, counts = np.unique(seg, return_counts=True)
    total = seg.size
    obj_px: dict = {}
    obj_cat: dict = {}
    empty = 0
    wheels = set()
    wheel_px: dict = {}
    for i, c in zip(ids, counts):
        lab = id_to_label.get(str(int(i)), "UNLABELLED")
        if isinstance(lab, dict):
            lab = lab.get("class", "") or str(lab)
        key, cat = classify(str(lab))
        if key is None:
            empty += int(c)
            continue
        obj_px[key] = obj_px.get(key, 0) + int(c)
        obj_cat[key] = cat
        if cat == "valve_handwheel":
            wheels.add(key)
            wheel_px[key] = wheel_px.get(key, 0) + int(c)
    vis = {k for k, v in obj_px.items() if v >= min_px}
    by_cat: dict = {}
    for k in vis:
        by_cat[obj_cat[k]] = by_cat.get(obj_cat[k], 0) + 1
    # valves with a visible handwheel: wheel prim visible (>= min_px)
    valves_wheel = sum(1 for k in wheels if wheel_px.get(k, 0) >= 20)
    foreground = sum(v for c, v in by_cat.items() if c not in ("background_plant",))
    primary = sum(v for c, v in by_cat.items() if c not in ("background_plant", "pipe_fitting", "pipe_support"))
    cats = by_cat
    return dict(
        distinct_visible_objects=len(vis), distinct_visible_objects_excluding_background_plant=int(foreground), distinct_visible_primary_objects_excl_fittings_supports_background=int(primary), by_category=cats,
        valves_with_visible_handwheel=int(valves_wheel),
        pipe_segments_visible=cats.get("pipe_segment", 0), small_instruments_visible=cats.get("gauge", 0) + cats.get("instrument_box", 0) + cats.get("enclosure", 0) + cats.get("control_valve", 0) * 0,
        cable_or_tubing_bundles_visible=cats.get("tubing_bundle", 0) + cats.get("cable_bundle", 0),
        empty_fraction=round(empty / total, 4), min_pixels_for_visible=min_px,
    )
