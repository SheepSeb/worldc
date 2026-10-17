"""Preview renderers (core deps only).  Photoreal RTX renders live in ``isaac_render`` (run under Isaac's python).

Lazy exports: ``python -m worldc.render.isaac_render`` must not import ``pxr`` before Isaac Sim has loaded it."""

__all__ = ["render_proxy", "render_topdown_map"]


def __getattr__(name):
    if name in __all__:
        from . import proxy

        return getattr(proxy, name)
    raise AttributeError(name)
