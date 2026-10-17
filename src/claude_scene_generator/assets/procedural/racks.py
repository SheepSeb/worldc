"""Selective pallet rack row (uprights, beam pairs per level, column guards).  Ported from ``Structures.racks``.

Local frame: the row runs along local +y; ``face`` = +1 means pallets are accessed from the +x side (the aisle lies at
+x), -1 from -x.  Every pallet position is a support anchor ``slot_s{seg}_b{bay}_p{pos}_L{level}`` whose meta carries
segment / bay / slot / level / level_z / face so ``fill`` placements and nav goals can address it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...geom import Q_IDENT
from ..base import Anchor, Asset, Built, Footprint, Params, PhysicsInfo
from ..registry import register

PAL_RECT_Y = [(-0.4, -0.6), (0.4, -0.6), (0.4, 0.6), (-0.4, 0.6)]  # EUR pallet footprint, long side along the row


class RackParams(Params):
    segments: list[tuple[float, float]] = Field([(-4.3, 1.1), (4.3, 9.7)], description="Rack segments as [y0, y1] along local y (m); gaps between segments are cross-aisles")
    bays_per_segment: int = Field(2, ge=1, le=20)
    bay_length: float = Field(2.7, ge=1.0, le=4.0, description="Upright frame spacing along the row (m)")
    slots_per_bay: int = Field(2, ge=1, le=4)
    levels: list[float] = Field([0.2, 1.8, 3.4, 5.0], description="Beam top heights (m); a pallet slot per level")
    depth: float = Field(0.9, ge=0.6, le=1.5, description="Frame depth across the row (m)")
    upright_height: float = Field(6.4, ge=1.0, le=15)
    face: Literal[1, -1] = Field(1, description="+1: aisle (pallet access) on local +x; -1: on local -x")
    upright_color: tuple[float, float, float] = Field((0.10, 0.25, 0.65), description="RGB 0-1")
    beam_color: tuple[float, float, float] = Field((0.95, 0.45, 0.08), description="RGB 0-1")
    guards: bool = Field(True, description="Yellow column protectors at both segment ends")


POST, BEAM_H, BEAM_W = 0.08, 0.10, 0.05


@register
class RackRow(Asset):
    id = "rack_row"
    description = ("Selective steel pallet-rack row: upright frames, beam pairs at each level, column guards. "
                   "Exposes one support anchor per pallet position (slot_s{seg}_b{bay}_p{pos}_L{level}).")
    tags = frozenset({"rack", "shelving", "storage", "static", "steel", "support", "warehouse"})
    Params = RackParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "racks", "asset": "rack_row", "count": 12, "placement": {"layout": {"type": "rack_aisles", "aisles": 6, "aisle_width": 3.2}}}

    def footprint(self, p):
        segs = [Footprint.rect(1.0, (yb - ya) + 0.7, p.upright_height, 0.0, (ya + yb) / 2) for ya, yb in p.segments]
        return Footprint.union(segs) if len(segs) > 1 else segs[0]

    def _frames(self, p, ya):
        return [ya + i * p.bay_length for i in range(p.bays_per_segment + 1)]

    def anchors(self, p):
        out = []
        for s, (ya, _) in enumerate(p.segments):
            fr = self._frames(p, ya)
            for b in range(p.bays_per_segment):
                for pi in range(p.slots_per_bay):
                    yc = fr[b] + (0.713 + pi * 1.273 if p.slots_per_bay == 2 else (pi + 0.5) * p.bay_length / p.slots_per_bay)
                    for li, z in enumerate(p.levels):
                        out.append(Anchor(f"slot_s{s}_b{b}_p{pi}_L{li}", "support", (0.0, yc, z), polygon=PAL_RECT_Y,
                                          meta={"segment": s, "bay": b, "slot": pi, "level": li, "level_z": z, "face": p.face}))
        return out

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        up_m = m.color(p.upright_color, 0.45, 0.5)
        bm_m = m.color(p.beam_color, 0.45, 0.5)
        steel = m.phys("steel")
        parts = []

        def add(kind, name, lpos, size, **extra):
            mat = {"upright": up_m, "beam": bm_m, "guard": m.get("safety_yellow")}[kind]
            k.cube(f"{path}/{name}", size, lpos, Q_IDENT, mat, steel, collide=True)
            wp, _ = pose.world(lpos)
            parts.append({"type": kind, "name": name, "position": [round(float(v), 3) for v in wp], "size": list(size), **extra})

        half = p.depth / 2 - 0.05
        beam_x = half - 0.065
        for s, (ya, yb) in enumerate(p.segments):
            fr = self._frames(p, ya)
            for fi, yf in enumerate(fr):
                for side, sx in (("F", p.face), ("B", -p.face)):
                    add("upright", f"Upright_s{s}_f{fi}_{side}", (sx * half, yf, p.upright_height / 2), (POST, POST, p.upright_height))
            for b in range(p.bays_per_segment):
                for li, z in enumerate(p.levels):
                    for side, sx in (("F", p.face), ("B", -p.face)):
                        add("beam", f"Beam_s{s}_b{b}_L{li}_{side}", (sx * beam_x, fr[b] + p.bay_length / 2, z - BEAM_H / 2),
                            (BEAM_W, p.bay_length - POST, BEAM_H), level=li, top_z=z)
            if p.guards:
                for end_y, d in ((ya - 0.2, -1), (yb + 0.2, 1)):
                    add("guard", f"Guard_s{s}_{'S' if d < 0 else 'N'}", (0.0, end_y, 0.25), (p.depth + 0.05, 0.12, 0.5))
        k.tag(k.stage.GetPrimAtPath(path), "rack_row")
        ctx.record(path, "rack_row", pose.pos, pose.quat, "static", friction=0.6, face=p.face, levels=list(p.levels),
                   n_uprights=sum(1 for q in parts if q["type"] == "upright"), n_beams=sum(1 for q in parts if q["type"] == "beam"),
                   parts=parts)
        anchors = [a.to_world(pose) for a in self.anchors(p)]
        return Built(path, anchors, info={"face": p.face, "n_slots": len(anchors)})
