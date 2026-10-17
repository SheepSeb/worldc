"""Thin authoring layer over pxr: primitives, materials, rigid bodies, joints, scene-graph records.

Only ``pxr`` core + ``UsdPhysics`` are used.  PhysX-specific schemas (``PhysxRigidBodyAPI`` ...) are
authored as plain applied-API names + attributes so the generator runs in a vanilla ``usd-core``
environment (no Isaac Sim needed); Isaac Sim's registered PhysxSchema picks them up on load
(verified by ``isaac_validate.py``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .geom import Part, Q_IDENT, composite_inertia, compose, qmat, qmul, yaw_of

T = Sdf.ValueTypeNames


def _v3(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _q(q):
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


class Kit:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)
        self.world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(self.world.GetPrim())
        self.records: list[dict] = []
        self.ns = "scene"  # attribute namespace used by tag() (specs may set output.namespace, e.g. "warehouse")
        self._mats: dict[str, UsdShade.Material] = {}
        self._pmats: dict[str, UsdShade.Material] = {}
        self._counters: dict[str, int] = {}
        UsdGeom.Scope.Define(self.stage, "/World/Looks")
        UsdGeom.Scope.Define(self.stage, "/World/PhysicsMaterials")

    # ------------------------------------------------------------------ naming
    def next_id(self, key: str) -> int:
        n = self._counters.get(key, 0)
        self._counters[key] = n + 1
        return n

    # ------------------------------------------------------------------ transforms
    @staticmethod
    def set_pose(prim, pos=(0, 0, 0), quat=Q_IDENT, scale=None):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(c) for c in pos]))
        if tuple(quat) != Q_IDENT:
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(_q(quat))
        if scale is not None:
            xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(_v3(scale))

    def xform(self, path, pos=(0, 0, 0), quat=Q_IDENT):
        x = UsdGeom.Xform.Define(self.stage, path)
        self.set_pose(x.GetPrim(), pos, quat)
        return x.GetPrim()

    def scope(self, path):
        return UsdGeom.Scope.Define(self.stage, path).GetPrim()

    # ------------------------------------------------------------------ attributes / schemas
    @staticmethod
    def attr(prim, name, typ, value, custom=True):
        a = prim.CreateAttribute(name, typ, custom)
        a.Set(value)
        return a

    @staticmethod
    def physx(prim, api: str, **attrs):
        """Apply a PhysX API schema by name and author its attributes (namespace given by schema)."""
        prim.AddAppliedSchema(api)
        for name, (typ, val) in attrs.items():
            prim.CreateAttribute(name, typ, False).Set(val)

    def tag(self, prim, category: str, **extra):
        self.attr(prim, f"{self.ns}:category", T.String, category)
        for k, v in extra.items():
            if isinstance(v, bool):
                self.attr(prim, f"{self.ns}:{k}", T.Bool, v)
            elif isinstance(v, int):
                self.attr(prim, f"{self.ns}:{k}", T.Int, v)
            elif isinstance(v, float):
                self.attr(prim, f"{self.ns}:{k}", T.Double, v)
            else:
                self.attr(prim, f"{self.ns}:{k}", T.String, str(v))

    # ------------------------------------------------------------------ materials
    def material(self, name, color=(0.5, 0.5, 0.5), rough=0.7, metal=0.0, tex=None, opacity=1.0, emissive=None):
        key = name
        if key in self._mats:
            return self._mats[key]
        mpath = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, mpath)
        sh = UsdShade.Shader.Define(self.stage, mpath + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", T.Float).Set(float(rough))
        sh.CreateInput("metallic", T.Float).Set(float(metal))
        if opacity < 1.0:
            sh.CreateInput("opacity", T.Float).Set(float(opacity))
        if emissive is not None:
            sh.CreateInput("emissiveColor", T.Color3f).Set(Gf.Vec3f(*emissive))
        if tex:
            rd = UsdShade.Shader.Define(self.stage, mpath + "/st")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", T.String).Set("st")
            rd.CreateOutput("result", T.Float2)
            tx = UsdShade.Shader.Define(self.stage, mpath + "/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", T.Asset).Set(Sdf.AssetPath(tex))
            tx.CreateInput("sourceColorSpace", T.Token).Set("sRGB")
            tx.CreateInput("wrapS", T.Token).Set("repeat")
            tx.CreateInput("wrapT", T.Token).Set("repeat")
            tx.CreateInput("st", T.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            tx.CreateOutput("rgb", T.Float3)
            sh.CreateInput("diffuseColor", T.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", T.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
        sh.CreateOutput("surface", T.Token)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self._mats[key] = mat
        return mat

    def phys_material(self, name, mu, restitution=0.05, combine=None):
        if name in self._pmats:
            return self._pmats[name]
        m = UsdShade.Material.Define(self.stage, f"/World/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(float(mu))
        api.CreateDynamicFrictionAttr(float(mu))
        api.CreateRestitutionAttr(float(restitution))
        api.CreateDensityAttr(0.0)
        if combine:
            self.physx(m.GetPrim(), "PhysxMaterialAPI", **{"physxMaterial:frictionCombineMode": (T.Token, combine)})
        self._pmats[name] = m
        return m

    @staticmethod
    def bind(prim, mat, purpose=None):
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        if purpose:
            api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, purpose)
        else:
            api.Bind(mat)

    # ------------------------------------------------------------------ primitives
    def _finish(self, prim, vmat, pmat, collide, guide):
        if vmat is not None:
            self.bind(prim, vmat)
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            if pmat is not None:
                self.bind(prim, pmat, "physics")
        if guide:
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    def cube(self, path, size, pos=(0, 0, 0), quat=Q_IDENT, vmat=None, pmat=None, collide=False, guide=False):
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        self.set_pose(c.GetPrim(), pos, quat, size)
        self._finish(c.GetPrim(), vmat, pmat, collide, guide)
        return c.GetPrim()

    def cylinder(self, path, radius, height, pos=(0, 0, 0), quat=Q_IDENT, vmat=None, pmat=None, collide=False, guide=False):
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.CreateAxisAttr("Z")
        c.CreateExtentAttr([(-radius, -radius, -height / 2), (radius, radius, height / 2)])
        self.set_pose(c.GetPrim(), pos, quat)
        self._finish(c.GetPrim(), vmat, pmat, collide, guide)
        return c.GetPrim()

    def capsule(self, path, radius, height, pos=(0, 0, 0), quat=Q_IDENT, vmat=None, pmat=None, collide=False):
        c = UsdGeom.Capsule.Define(self.stage, path)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.CreateAxisAttr("Z")
        h2 = height / 2 + radius
        c.CreateExtentAttr([(-radius, -radius, -h2), (radius, radius, h2)])
        self.set_pose(c.GetPrim(), pos, quat)
        self._finish(c.GetPrim(), vmat, pmat, collide, False)
        return c.GetPrim()

    def sphere(self, path, radius, pos=(0, 0, 0), vmat=None, pmat=None, collide=False):
        c = UsdGeom.Sphere.Define(self.stage, path)
        c.CreateRadiusAttr(float(radius))
        c.CreateExtentAttr([(-radius,) * 3, (radius,) * 3])
        self.set_pose(c.GetPrim(), pos)
        self._finish(c.GetPrim(), vmat, pmat, collide, False)
        return c.GetPrim()

    def cone(self, path, radius, height, pos=(0, 0, 0), quat=Q_IDENT, vmat=None):
        c = UsdGeom.Cone.Define(self.stage, path)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.CreateAxisAttr("Z")
        c.CreateExtentAttr([(-radius, -radius, -height / 2), (radius, radius, height / 2)])
        self.set_pose(c.GetPrim(), pos, quat)
        self._finish(c.GetPrim(), vmat, None, False, False)
        return c.GetPrim()

    def mesh_box(self, path, size, pos=(0, 0, 0), quat=Q_IDENT, vmat=None, uv=True):
        """Visual-only box mesh with per-face UVs (so cardboard textures/labels map on every face)."""
        sx, sy, sz = [s / 2 for s in size]
        P = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
             (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
        normals_f = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        pts, nrm = [], []
        for f, n in zip(faces, normals_f):
            for i in f:
                pts.append(P[i])
                nrm.append(n)
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        m.CreateFaceVertexCountsAttr([4] * 6)
        m.CreateFaceVertexIndicesAttr(list(range(24)))
        m.CreateExtentAttr([(-sx, -sy, -sz), (sx, sy, sz)])
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if uv:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", T.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in [(0, 0), (0, 1), (1, 1), (1, 0)] * 6]))
        self.set_pose(m.GetPrim(), pos, quat)
        if vmat is not None:
            self.bind(m.GetPrim(), vmat)
        return m.GetPrim()

    def mesh(self, path, points, faces_counts, faces_idx, uvs=None, vmat=None, pos=(0, 0, 0), quat=Q_IDENT):
        m = UsdGeom.Mesh.Define(self.stage, path)
        pts = np.asarray(points, float)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        m.CreateFaceVertexCountsAttr(list(faces_counts))
        m.CreateFaceVertexIndicesAttr(list(faces_idx))
        m.CreateExtentAttr([tuple(pts.min(0)), tuple(pts.max(0))])
        m.CreateSubdivisionSchemeAttr("none")
        if uvs is not None:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", T.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in uvs]))
        self.set_pose(m.GetPrim(), pos, quat)
        if vmat is not None:
            self.bind(m.GetPrim(), vmat)
        return m.GetPrim()

    # ------------------------------------------------------------------ physics
    def physics_scene(self, path="/World/PhysicsScene", hz=120):
        sc = UsdPhysics.Scene.Define(self.stage, path)
        sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr(9.81)
        self.physx(
            sc.GetPrim(), "PhysxSceneAPI",
            **{
                "physxScene:solverType": (T.Token, "TGS"),
                "physxScene:timeStepsPerSecond": (T.UInt, hz),
                "physxScene:enableCCD": (T.Bool, False),
                "physxScene:enableStabilization": (T.Bool, True),
                "physxScene:enableGPUDynamics": (T.Bool, False),
                "physxScene:broadphaseType": (T.Token, "MBP"),
                "physxScene:bounceThreshold": (T.Float, 0.5),
                "physxScene:frictionOffsetThreshold": (T.Float, 0.01),
                "physxScene:frictionCorrelationDistance": (T.Float, 0.025),
            },
        )
        return sc.GetPrim()

    def rigid_body(self, path, pos, quat, parts: list[Part] | None = None, mass=1.0, kinematic=False,
                   disable_gravity=False, iters=(12, 2), linear_damping=0.0, angular_damping=0.05,
                   com_inertia=None, max_depen=1.0):
        """Create an Xform rigid body with explicit mass / COM / diagonal inertia / principal axes."""
        p = self.xform(path, pos, quat)
        UsdPhysics.RigidBodyAPI.Apply(p)
        if kinematic:
            UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
        if com_inertia is None:
            com, diag, axes, M = composite_inertia(parts, mass)
        else:
            com, diag, axes = com_inertia
        mapi = UsdPhysics.MassAPI.Apply(p)
        mapi.CreateMassAttr(float(mass))
        mapi.CreateCenterOfMassAttr(_v3(com))
        mapi.CreateDiagonalInertiaAttr(_v3(diag))
        mapi.CreatePrincipalAxesAttr(_q(axes))
        a = {
            "physxRigidBody:solverPositionIterationCount": (T.Int, iters[0]),
            "physxRigidBody:solverVelocityIterationCount": (T.Int, iters[1]),
            "physxRigidBody:maxDepenetrationVelocity": (T.Float, max_depen),
            "physxRigidBody:linearDamping": (T.Float, linear_damping),
            "physxRigidBody:angularDamping": (T.Float, angular_damping),
        }
        if disable_gravity:
            a["physxRigidBody:disableGravity"] = (T.Bool, True)
        self.physx(p, "PhysxRigidBodyAPI", **a)
        return p

    def static_collider(self, prim, pmat):
        UsdPhysics.CollisionAPI.Apply(prim)
        self.bind(prim, pmat, "physics")

    def joint(self, path, kind, body0, body1, pos0, pos1, axis="Z", lower=0.0, upper=1.0,
              target=0.0, stiffness=1e4, damping=1e3, max_force=1e6, rot0=Q_IDENT, rot1=Q_IDENT, friction=0.0):
        """Revolute (degrees) or prismatic (metres) joint with a position drive; body0 may be a static prim."""
        if kind == "revolute":
            j = UsdPhysics.RevoluteJoint.Define(self.stage, path)
            drive = "angular"
        else:
            j = UsdPhysics.PrismaticJoint.Define(self.stage, path)
            drive = "linear"
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr(_v3(pos0))
        j.CreateLocalRot0Attr(_q(rot0))
        j.CreateLocalPos1Attr(_v3(pos1))
        j.CreateLocalRot1Attr(_q(rot1))
        j.CreateAxisAttr(axis)
        if lower is not None:  # None -> unlimited (continuous revolute)
            j.CreateLowerLimitAttr(float(lower))
            j.CreateUpperLimitAttr(float(upper))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive)
        d.CreateTypeAttr("force")
        d.CreateTargetPositionAttr(float(target))
        d.CreateStiffnessAttr(float(stiffness))
        d.CreateDampingAttr(float(damping))
        d.CreateMaxForceAttr(float(max_force))
        a = {"physxJoint:jointFriction": (T.Float, float(friction))}
        self.physx(j.GetPrim(), "PhysxJointAPI", **a)
        return j.GetPrim()

    def fixed_base(self, path, world_pos, world_quat):
        """Tiny articulation-root link welded to the world (fixed joint with empty body0)."""
        from .geom import Part

        b = self.rigid_body(path, (0, 0, 0), Q_IDENT, parts=[Part("box", (0.02, 0.02, 0.02))], mass=1.0, disable_gravity=True)
        fj = UsdPhysics.FixedJoint.Define(self.stage, path + "_WorldFix")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        fj.CreateLocalPos0Attr(_v3(world_pos))
        fj.CreateLocalRot0Attr(_q(world_quat))
        fj.CreateLocalPos1Attr(_v3((0, 0, 0)))
        fj.CreateLocalRot1Attr(_q(Q_IDENT))
        self.articulation_root(b)
        return b

    def articulation_root(self, prim):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        self.physx(prim, "PhysxArticulationAPI", **{
            "physxArticulation:solverPositionIterationCount": (T.Int, 32),
            "physxArticulation:solverVelocityIterationCount": (T.Int, 4),
            "physxArticulation:enabledSelfCollisions": (T.Bool, False),
        })

    # ------------------------------------------------------------------ scene-graph records
    def record(self, path, category, pos, quat, body_type, size=None, mass=None, friction=None,
               articulation=None, zone=None, **extra):
        r = {
            "path": str(path),
            "name": Sdf.Path(str(path)).name,
            "category": category,
            "body_type": body_type,
            "position": [round(float(c), 4) for c in pos],
            "orientation_wxyz": [round(float(c), 5) for c in quat],
            "yaw_deg": round(math.degrees(yaw_of(quat)), 2),
        }
        if size is not None:
            r["size"] = [round(float(c), 4) for c in size]
        if mass is not None:
            r["mass_kg"] = round(float(mass), 4)
        if friction is not None:
            r["friction"] = friction
        r["articulation"] = articulation
        if zone:
            r["zone"] = zone
        r.update(extra)
        self.records.append(r)
        return r

    def save(self):
        self.stage.GetRootLayer().Save()
