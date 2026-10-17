"""Shared fixtures.  The whole session refuses to run without an Isaac Sim install (project policy, docs/PLAN.md §8)."""
from __future__ import annotations

import numpy as np
import pytest

from claude_scene_generator.isaac_env import IsaacSimUnavailable, require_isaac_sim


def pytest_sessionstart(session):
    try:
        require_isaac_sim()
    except IsaacSimUnavailable as e:
        pytest.exit(f"Isaac Sim gate failed: {e.issue['message']} {e.issue['hint']}", returncode=2)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running (full warehouse builds over many seeds)")


@pytest.fixture
def scratch_ctx(tmp_path):
    """A BuildContext writing to a fresh stage in tmp_path (namespace ``scene``)."""
    from claude_scene_generator.assets.base import BuildContext
    from claude_scene_generator.materials import Materials, TextureCache
    from claude_scene_generator.usdkit import Kit

    kit = Kit(tmp_path / "scene.usd")
    kit.ns = "scene"
    kit.physics_scene()
    ctx = BuildContext(kit, Materials(kit), TextureCache(tmp_path, seed=0), seed=0, out_dir=tmp_path)
    return ctx


def build_asset(ctx, asset_id, params=None, pose=None, seed=0, path=None):
    """Build one asset instance into ``ctx``; returns (asset, params, Built)."""
    from claude_scene_generator.assets import Pose, get

    a = get(asset_id)
    p = a.params(params)
    path = path or ctx.alloc("/World/Test", asset_id.replace("/", "_"))
    built = a.build(ctx, path, pose or Pose(), p, np.random.default_rng(seed))
    return a, p, built
