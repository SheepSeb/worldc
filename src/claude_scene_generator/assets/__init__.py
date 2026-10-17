"""Asset registry: procedural builders, mesh assets, catalog and LLM-generated assets behind one interface."""
from .base import Anchor, Asset, BuildContext, Built, Environment, Footprint, JointInfo, Keepout, Params, PhysicsInfo, Pose
from .registry import describe, get, ids, list_assets, register

__all__ = ["Anchor", "Asset", "BuildContext", "Built", "Environment", "Footprint", "JointInfo", "Keepout", "Params",
           "PhysicsInfo", "Pose", "describe", "get", "ids", "list_assets", "register"]
