"""Play an Isaac Sim animation clip on a referenced Isaac Sim character (UsdSkel).  Dependency-free (pxr only) so it also runs under
Isaac's python.

NVIDIA's clips (``Isaac/People/Animations/*.skelanim.usd``) are authored for the 81-joint ``biped_demo`` rig, not for the characters' own
101-joint rig, so Isaac Sim retargets them at runtime with an animation graph.  Here the retarget is baked instead: both rigs are in a
T-pose, so every mapped bone is rotated by the same world-space change it makes in the clip (``D = anim * bind^-1``, re-expressed in the
character's axes) and the result is converted back to local rotations.  Bones the clip does not drive (twist, facial, ...) keep their rest
pose relative to their parent.  The clip's root motion (travel, heading) is dropped because ``runtime.PathFollower`` moves the body; the
pelvis keeps its vertical bob, scaled to the character's size.
"""
from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdSkel, Vt

# clip joint -> character joint (short names).  Spine1..Chest land on the character's three spine bones; Spine3 and the fourth finger
# segment have no counterpart.
CLIP_TO_CHAR = {"Pelvis": "Hip", "Spine1": "Waist", "Spine2": "Spine01", "Chest": "Spine02", "Neck1": "NeckTwist01", "Neck2": "NeckTwist02", "Head": "Head"}
for _s, _a in (("R", "R"), ("L", "L")):
    CLIP_TO_CHAR.update({f"{_s}_UpLeg": f"{_a}_Thigh", f"{_s}_LoLeg": f"{_a}_Calf", f"{_s}_Ankle": f"{_a}_Foot", f"{_s}_Ball": f"{_a}_ToeBase",
                         f"{_s}_Clavicle": f"{_a}_Clavicle", f"{_s}_UpArm": f"{_a}_Upperarm", f"{_s}_LoArm": f"{_a}_Forearm", f"{_s}_Wrist": f"{_a}_Hand"})
# the clips' relaxed hands (the character's rest pose is a flat, open T-pose hand)
CLIP_TO_CHAR_FINGERS = {f"{c}_{n}Finger{i}": f"{c}_{m}{i}" for c in "LR" for n, m in (("Thumb", "Thumb"), ("Index", "Index"), ("Middle", "Mid"),
                                                                                   ("Ring", "Ring"), ("Pinky", "Pinky")) for i in (1, 2, 3)}
# the clips are Y-up facing +Z, the characters Z-up facing -Y: x stays, y -> z, z -> -y (a quarter turn about X)
AXES = Gf.Quatd(math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0)


class Rig:
    """A joint hierarchy with rest local transforms.  ``joints`` are USD joint tokens, parents before children; ``rest`` rows are
    [tx, ty, tz, qw, qx, qy, qz] (the tables in ``library/people``)."""

    def __init__(self, joints, rest):
        self.joints = list(joints)
        index = {j: i for i, j in enumerate(self.joints)}
        self.parent = [index[j.rsplit("/", 1)[0]] if "/" in j else -1 for j in self.joints]
        assert all(p < i for i, p in enumerate(self.parent)), "joints must list parents first"
        self.short = [j.rsplit("/", 1)[-1] for j in self.joints]
        self.by_short = {s: i for i, s in enumerate(self.short)}
        self.t = [Gf.Vec3d(*r[:3]) for r in rest]
        self.q = [Gf.Quatd(*r[3:]) for r in rest]

    def global_rots(self, local):
        out = []
        for i, q in enumerate(local):
            out.append(q if self.parent[i] < 0 else out[self.parent[i]] * q)
        return out

    def global_pos(self, local_q, local_t):
        rots, pos = self.global_rots(local_q), []
        for i, t in enumerate(local_t):
            pos.append(Gf.Vec3d(t) if self.parent[i] < 0 else pos[self.parent[i]] + Gf.Rotation(rots[self.parent[i]]).TransformDir(t))
        return pos


def _conj(q: Gf.Quatd) -> Gf.Quatd:
    return Gf.Quatd(q.GetReal(), -q.GetImaginary())


def _slerp(a: Gf.Quatd, b: Gf.Quatd, u: float) -> Gf.Quatd:
    if Gf.Dot(a, b) < 0:
        b = Gf.Quatd(-b.GetReal(), -b.GetImaginary())
    return Gf.Slerp(u, a, b)


class Retarget:
    """Clip rig -> character rig.  ``frame(rots, pelvis_t)`` turns one clip frame (local rotations of the clip's joints, the pelvis'
    local translation) into the character's local rotations / translations for ``self.names`` (the joints that move)."""

    def __init__(self, clip_rig: Rig, char_rig: Rig, fingers: bool = True):
        pairs = {**CLIP_TO_CHAR, **(CLIP_TO_CHAR_FINGERS if fingers else {})}
        self.clip, self.char = clip_rig, char_rig
        self.pairs = [(clip_rig.by_short[c], char_rig.by_short[t]) for c, t in pairs.items() if c in clip_rig.by_short and t in char_rig.by_short]
        self.target_of = {t: c for c, t in self.pairs}
        self.order = sorted(self.target_of)  # character joints that move, parents first
        self.clip_bind = clip_rig.global_rots(clip_rig.q)
        self.char_bind = char_rig.global_rots(char_rig.q)
        self.axes_inv = _conj(AXES)
        hip, pelvis = char_rig.by_short["Hip"], clip_rig.by_short["Pelvis"]
        self.hip = hip
        self.pelvis = pelvis
        # the character's pelvis height over the clip's scales the vertical bob; the hip moves in its parent's frame
        self.bob_scale = char_rig.global_pos(char_rig.q, char_rig.t)[hip][2] / clip_rig.global_pos(clip_rig.q, clip_rig.t)[pelvis][1]
        self.hip_parent_inv = _conj(self.char_bind[char_rig.parent[hip]]) if char_rig.parent[hip] >= 0 else Gf.Quatd(1, 0, 0, 0)

    def frame(self, clip_rots, pelvis_t):
        clip, char = self.clip, self.char
        local = list(clip_rots)
        local[0] = clip.q[0]  # drop the clip's root heading
        anim = clip.global_rots(local)
        out_q = list(char.q)
        char_global: list[Gf.Quatd] = []
        for i in range(len(char.joints)):
            p = char.parent[i]
            if i in self.target_of:
                c = self.target_of[i]
                delta = anim[c] * _conj(self.clip_bind[c])  # world-space change of this bone in the clip
                g = AXES * delta * self.axes_inv * self.char_bind[i]
                out_q[i] = g if p < 0 else _conj(char_global[p]) * g
            else:
                g = char.q[i] if p < 0 else char_global[p] * char.q[i]
            char_global.append(g)
        d = Gf.Vec3d(pelvis_t) - Gf.Vec3d(self.clip.t[self.pelvis])
        shift = Gf.Rotation(self.hip_parent_inv * AXES).TransformDir(d * self.bob_scale)
        return [out_q[i] for i in self.order], char.t[self.hip] + shift


def read_clip(stage, prim_path="/Root"):
    """(joint tokens, per-frame local rotations [Gf.Quatd], per-frame translations [Gf.Vec3d], start_code, tcps) of a SkelAnimation."""
    anim = UsdSkel.Animation(stage.GetPrimAtPath(prim_path))
    joints = [str(j) for j in anim.GetJointsAttr().Get()]
    rot, trn = anim.GetRotationsAttr(), anim.GetTranslationsAttr()
    times = rot.GetTimeSamples()
    rots = [[Gf.Quatd(q) for q in rot.Get(t)] for t in times]
    trans = [[Gf.Vec3d(v) for v in trn.Get(t)] for t in times]
    return joints, rots, trans, times, stage.GetTimeCodesPerSecond() or 30.0


def blend(a, b, w: float):
    """Blend two retargeted poses ((rotations, hip translation) as ``Retarget.frame`` returns them); w = 0 is a, 1 is b."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    return [_slerp(x, y, w) for x, y in zip(a[0], b[0])], a[1] * (1 - w) + b[1] * w


def author_samples(stage, skeleton_path: str, rt: Retarget, poses: list, step_code: float, total_code: float, name: str = "Walk"):
    """Author ``poses`` (one period of retargeted poses, ``step_code`` time codes apart) as a SkelAnimation on ``skeleton_path``,
    repeated until ``total_code``.  Identical samples are stored once by the crate file format, so a long timeline costs one period."""
    char = rt.char
    skel = stage.OverridePrim(skeleton_path)
    anim_path = f"{skeleton_path}/{name}"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr(Vt.TokenArray([char.joints[i] for i in rt.order]))
    anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(rt.order)))
    rest_t = Vt.Vec3fArray([Gf.Vec3f(char.t[i]) for i in rt.order])
    hip = rt.order.index(rt.hip)
    packed = []
    for q, hip_t in poses:
        t = Vt.Vec3fArray(rest_t)
        t[hip] = Gf.Vec3f(hip_t)
        packed.append((Vt.QuatfArray([Gf.Quatf(x) for x in q]), t))
    rot_attr, trn_attr = anim.CreateRotationsAttr(), anim.CreateTranslationsAttr()
    for k in range(int(total_code / step_code) + 1):
        q, t = packed[k % len(packed)]
        rot_attr.Set(q, k * step_code)
        trn_attr.Set(t, k * step_code)
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])
    return anim
