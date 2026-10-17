import json

import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from conftest import build_asset
from worldc.assets import Pose, describe, get
from worldc.colliders import collect


@pytest.mark.parametrize("variant", ["boxes", "wrapped", "totes", "empty"])
def test_pallet_load_variants(scratch_ctx, variant):
    a, p, b = build_asset(scratch_ctx, "pallet_load", {"variant": variant, "full": True}, Pose.xy_yaw(1, 2, 0.3), seed=3)
    assert b.info["kind"] == variant
    top = b.anchor("top").pos[2]
    assert top >= b.anchor("deck").pos[2]
    if variant in ("boxes", "wrapped"):
        assert b.info["n_boxes"] > 0 and top > 0.3
    cats = {r["category"] for r in scratch_ctx.kit.records}
    assert "pallet" in cats


def test_deterministic(tmp_path_factory, scratch_ctx):
    _, _, b1 = build_asset(scratch_ctx, "pallet_load", {"variant": "boxes"}, seed=7)
    n1 = len(scratch_ctx.kit.records)
    recs1 = [r["position"] for r in scratch_ctx.kit.records]
    _, _, b2 = build_asset(scratch_ctx, "pallet_load", {"variant": "boxes"}, seed=7)
    recs2 = [r["position"] for r in scratch_ctx.kit.records[n1:]]
    assert recs1 == recs2


def test_box_and_jack_physics(scratch_ctx):
    build_asset(scratch_ctx, "cardboard_box", {"size": [0.5, 0.4, 0.3]})
    build_asset(scratch_ctx, "pallet_jack_tipped", pose=Pose.xy_yaw(3, 0))
    build_asset(scratch_ctx, "tote", {"n_items": 5}, Pose.xy_yaw(-3, 0))
    st = scratch_ctx.kit.stage
    for prim in st.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            assert m.GetMassAttr().Get() > 0
            assert all(v > 0 for v in m.GetDiagonalInertiaAttr().Get())


def test_describe_json(scratch_ctx):
    for aid in ("pallet", "pallet_load", "cardboard_box", "tote", "pallet_jack_tipped"):
        json.dumps(describe(aid))


def test_param_errors():
    from worldc.assets.base import ParamError

    with pytest.raises(ParamError) as e:
        get("pallet_load").params({"variant": "crates"})
    assert "variant" in str(e.value)
    with pytest.raises(ParamError):
        get("pallet_load").params({"lenght": 1})
    p = get("pallet_load").params({"variant": {"choice": ["boxes", "totes"]}}, rng=np.random.default_rng(0))
    assert p.variant in ("boxes", "totes")
