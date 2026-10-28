"""Stand-in for `claude -p ... --output-format stream-json`: emits the same event shapes, but the 'model' is a fixed script and the
tool calls are executed for real through worldc.tools.  Behaviour via FAKE_MODE: ok | invalid | nospec | crash."""
import base64
import json
import os
import sys
import uuid

from worldc import tools as T

MODE = os.environ.get("FAKE_MODE", "ok")
MINI = {"environment": {"footprint": {"size": [6, 6]}, "walls": {"openings": [{"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}]}}, "zones": [{"id": "z", "rect": [-2.5, -2.5, 2.5, 2.5]}],
        "objects": [{"id": "pallets", "asset": "pallet", "count": 2, "placement": {"inside": "z"}}]}
BAD = {**MINI, "objects": [{"id": "p", "asset": "palet", "placement": {"inside": "z"}}]}


def emit(**e):
    print(json.dumps(e), flush=True)


def call(name, inp):
    tid = "toolu_" + uuid.uuid4().hex[:10]
    emit(type="assistant", message={"id": "msg_" + tid, "role": "assistant", "model": "fake-model", "usage": {"input_tokens": 3, "output_tokens": 5},
                                    "content": [{"type": "tool_use", "id": tid, "name": "mcp__worldc__" + name, "input": inp, "caller": {"type": "direct"}}]})
    res = T.TOOLS[name](**inp)
    content = [{"type": "text", "text": json.dumps(res)}]
    img = res.get("result", {}).get("image")
    if img:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(open(img, "rb").read()).decode()}})
    emit(type="user", message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]})
    return res


emit(type="system", subtype="init", session_id="sess-1", model="fake-model", tools=["mcp__worldc__build_scene"], mcp_servers=[{"name": "worldc", "status": "connected"}])
if MODE == "crash":
    sys.exit(3)
if MODE == "invalid":
    call("validate_spec", {"spec": BAD})
elif MODE != "nospec":
    call("search_assets", {"query": "pallet", "limit": 2})
    call("validate_spec", {"spec": BAD})  # a failed attempt first, as real runs have
    call("validate_spec", {"spec": MINI})
    b = call("build_scene", {"spec": MINI, "seed": 0})
    call("render", {"path": b["result"]["out_dir"], "camera": "topdown_map", "engine": "proxy"})
emit(type="assistant", message={"id": "msg_final", "role": "assistant", "model": "fake-model", "usage": {"input_tokens": 3, "output_tokens": 9},
                                "content": [{"type": "thinking", "thinking": "", "signature": "xyz"}, {"type": "text", "text": "Built a 6x6 room with two pallets."}]})
emit(type="result", subtype="success", is_error=False, stop_reason="end_turn", num_turns=5, total_cost_usd=0.123, duration_ms=10,
     usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 10}, result="Built a 6x6 room with two pallets.")
