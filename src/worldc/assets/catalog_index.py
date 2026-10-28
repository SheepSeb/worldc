"""Offline index + search over an Isaac Sim / Omniverse asset root (PLAN.md §2.6).

    python -m worldc.assets.catalog_index index [--root URL|DIR] [--out FILE] [--packs Props Environments]
    python -m worldc.assets.catalog_index search "hand truck" [-k 8]

The index is a JSONL file (first line ``{"_meta": ...}``, then one asset per line) checked in under
``assets/library/catalog/``.  Search is local BM25 over name / folder / semantic-label tokens, so specs, tests and data
collection need no network.  Only assets themselves are fetched (and referenced by URL), never copied into outputs.

Indexing lists the root (S3 ListObjectsV2 or a local directory walk), drops thumbnails / materials / sublayer
configuration, collapses numbered instances of the same prop (``SM_CardBoxD_04_1169`` -> ``SM_CardBoxD_04`` x N) and,
unless ``--no-deep``, downloads each root layer (plus the layers it composes) to read bounds in metres (Z-up), authored
physics and semantic labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

DEFAULT_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
LIBRARY = Path(__file__).parent / "library" / "catalog"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldc" / "catalog_index"
USD_EXT = (".usd", ".usda", ".usdc")
SKIP_DIRS = {".thumbs", "materials", "textures", "configuration", "rivermark", "animations", ".asset_hashes", "parts", "meshes",
             "sublayers", "payloads", "variants", "xlsx"}
DEFAULT_PACKS = ("Props", "Environments", "People")
MAX_FILE = 80e6
STOP = {"sm", "usd", "usda", "usdc", "the", "a", "of", "for", "and", "with", "to", "in"}
SYNONYMS = {  # query word -> extra index words it should also match
    "trolley": ["cart", "dolly"], "handtruck": ["dolly"], "forklift": ["lift", "truck"], "crate": ["box", "carton"],
    "carton": ["box", "cardbox"], "cardboard": ["cardbox", "box"], "bin": ["tote", "klt"], "tote": ["bin", "klt"],
    "shelf": ["rack", "shelving"], "shelves": ["rack", "shelving"], "person": ["character", "worker", "human", "people"], "people": ["character", "worker", "human"],
    "worker": ["character", "human"], "table": ["desk"], "stand": ["rack"], "barrel": ["drum"],
}


# ------------------------------------------------------------------------------------------------------- listing
def _http_list(root: str, sub: str = ""):
    """Yield (path relative to ``root``, size, etag) for every object below ``root``/``sub`` on an S3-style HTTP root."""
    u = urllib.parse.urlparse(root)
    bucket_url, base = f"{u.scheme}://{u.netloc}/", u.path.lstrip("/").rstrip("/") + "/"
    prefix, tok = base + sub, None
    while True:
        q = {"list-type": "2", "max-keys": "1000", "prefix": prefix}
        if tok:
            q["continuation-token"] = tok
        x = urllib.request.urlopen(bucket_url + "?" + urllib.parse.urlencode(q), timeout=60).read().decode()
        for block in x.split("<Contents>")[1:]:
            key, size = re.search(r"<Key>(.*?)</Key>", block), re.search(r"<Size>(\d+)</Size>", block)
            etag = re.search(r"<ETag>(?:&quot;|\")?(.*?)(?:&quot;|\")?</ETag>", block)
            if key and size:
                yield key[1][len(base):], int(size[1]), etag[1] if etag else ""
        t = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not t:
            return
        tok = t[1]


def list_root(root: str, subs=("",)):
    """Yield (relative path, size, etag) below each sub-folder of ``root`` (S3-style HTTP root or a local directory)."""
    for sub in subs:
        if root.startswith(("http://", "https://")):
            yield from _http_list(root, sub)
            continue
        base = Path(root)
        for p in sorted((base / sub).rglob("*")) if (base / sub).is_dir() else []:
            if p.is_file():
                yield p.relative_to(base).as_posix(), p.stat().st_size, ""


# ------------------------------------------------------------------------------------------------------- candidates
def pack_of(rel: str) -> str:
    parts = rel.split("/")
    return parts[1] if parts[0] == "Isaac" and len(parts) > 1 else parts[0]


LICENSES = {
    "Environments": "limited-use",  # environment-supplement-LICENSE.txt: use only, no modification, no redistribution grant
}


def kind_of(e: dict) -> str:
    """``scene`` for whole environments (Isaac/Environments/<env>/<scene>.usd), ``asset`` for anything placeable."""
    parts = e["path"].split("/")
    return "scene" if pack_of(e["path"]) == "Environments" and "Props" not in parts else "asset"


def license_of(rel: str) -> str:
    """License class of a catalog path: ``limited-use`` (Isaac/Environments) or ``omniverse-content`` (everything else;
    individual prop folders may add third-party terms in their own README, e.g. Beaker is CC-BY 4.0)."""
    return LICENSES.get(pack_of(rel), "omniverse-content")


def _is_candidate(rel: str, packs) -> bool:
    parts = rel.split("/")
    if not rel.lower().endswith(USD_EXT) or (packs and pack_of(rel) not in packs):
        return False
    return not any(p.lower() in SKIP_DIRS for p in parts[:-1])


def _family(rel: str) -> str:
    stem, ext = os.path.splitext(rel)
    return re.sub(r"_\d{3,}$", "", stem) + ext  # numbered scene instances share one family


def _tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    toks = (t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in re.split(r"[^A-Za-z0-9]+", text.lower()))
    return [t for t in toks if t and t not in STOP]  # crude plural stemming: characters -> character


def _with_compounds(toks: list[str]) -> list[str]:
    """Tokens plus adjacent joins: "card box" also answers "cardbox"."""
    return toks + [a + b for a, b in zip(toks, toks[1:])]


def candidates(entries, packs=DEFAULT_PACKS):
    """Collapse listing entries to one record per asset family."""
    entries = list(entries)
    thumbs = {rel for rel, _, _ in entries if ".thumbs" in rel and rel.lower().endswith(".png")}
    fam: dict[str, dict] = {}
    for rel, size, etag in sorted(entries):
        if not _is_candidate(rel, packs):
            continue
        f = fam.setdefault(_family(rel), {"path": rel, "bytes": size, "etag": etag, "instances": 0})
        f["instances"] += 1
    out = []
    for f in fam.values():
        d, name = f["path"].rsplit("/", 1) if "/" in f["path"] else ("", f["path"])
        stem = name.rsplit(".", 1)[0]
        thumb = next((t for t in (f"{d}/.thumbs/256x256/{name}.png", f"{d}/.thumbs/256x256/{stem}.png") if t in thumbs), None)
        out.append({**f, "name": stem, "folder": d, **({"thumb": thumb} if thumb else {})})
    return out


# ------------------------------------------------------------------------------------------------------- deep pass
def _fetch(root: str, rel: str, cache: Path) -> Path:
    dst = cache / rel
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if root.startswith(("http://", "https://")):
        url = root.rstrip("/") + "/" + urllib.parse.quote(rel)
        with urllib.request.urlopen(url, timeout=60) as r:
            if int(r.headers.get("Content-Length") or 0) > MAX_FILE:
                raise OSError(f"{rel}: larger than {MAX_FILE / 1e6:.0f} MB")
            with tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(Path(root) / rel, tmp)
    tmp.rename(dst)
    return dst


def _mirror(root: str, rel: str, cache: Path, depth=0, seen=None, limit=60) -> Path:
    """Fetch ``rel`` and the layers it composes (sublayers / references / payloads) into ``cache``."""
    from pxr import Sdf

    seen = set() if seen is None else seen
    local = _fetch(root, rel, cache)
    seen.add(rel)
    if depth >= 6:
        return local
    layer = Sdf.Layer.FindOrOpen(str(local))
    for dep in (urllib.parse.unquote(d) for d in (layer.GetCompositionAssetDependencies() if layer else [])):
        if "://" in dep or not dep.lower().endswith(USD_EXT) or len(seen) >= limit:
            continue
        sub = os.path.normpath(os.path.join(os.path.dirname(rel), dep.lstrip("/") if dep.startswith("/") else dep)).replace("\\", "/")
        if sub.startswith("..") or sub in seen:
            continue
        try:
            _mirror(root, sub, cache, depth + 1, seen, limit)
        except OSError:
            pass  # a missing optional layer just leaves that part out of the bounds
    return local


def describe(root: str, rel: str, cache: Path) -> dict:
    """Bounds (m, Z-up), authored physics and semantic labels of one asset; ``{"deep": False, ...}`` if unreadable."""
    from pxr import Usd, UsdGeom

    try:
        local = _mirror(root, rel, cache / "mirror")
        stage = Usd.Stage.Open(str(local))
        if stage is None or not stage.GetDefaultPrim():
            return {"deep": False, "note": "no default prim"}
        mpu, up = UsdGeom.GetStageMetersPerUnit(stage) or 1.0, UsdGeom.GetStageUpAxis(stage)
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(
            stage.GetDefaultPrim()).ComputeAlignedRange()
        size = None
        if not rng.IsEmpty():
            s = [float(v) * mpu for v in rng.GetSize()]
            size = [s[0], s[2], s[1]] if up == "Y" else s
            if not all(math.isfinite(v) for v in size):
                size = None
        phys, labels, n_mesh = Counter(), set(), 0
        for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
            schemas = set(p.GetAppliedSchemas())
            phys["rigid_body"] += "PhysicsRigidBodyAPI" in schemas
            phys["collision"] += "PhysicsCollisionAPI" in schemas
            phys["articulation"] += "PhysicsArticulationRootAPI" in schemas
            phys["joints"] += p.GetTypeName().startswith("Physics") and "Joint" in p.GetTypeName()
            n_mesh += p.IsA(UsdGeom.Mesh)
            for a in p.GetAttributes():
                if a.GetName().startswith("semantic:") and a.GetName().endswith(":semanticData"):
                    v = a.Get()
                    if isinstance(v, str) and v:
                        labels.add(v.lower())
        return {"deep": True, "size_m": [round(v, 4) for v in size] if size else None, "up_axis": up, "meters_per_unit": mpu,
                "physics": {k: bool(v) for k, v in phys.items()}, "labels": sorted(labels), "meshes": n_mesh}
    except Exception as e:  # noqa: BLE001 - one bad asset must not abort the whole index
        return {"deep": False, "note": f"{type(e).__name__}: {str(e)[:120]}"}


def build_index(root: str = DEFAULT_ROOT, out: Path | str | None = None, packs=DEFAULT_PACKS, deep: bool = True,
                workers: int = 12, limit: int | None = None, cache: Path = CACHE, log=print) -> Path:
    cands = candidates(list_root(root, [f"Isaac/{p}/" for p in packs]), packs)
    if limit:
        cands = cands[:limit]
    log(f"{len(cands)} asset families from {root}")
    if deep:
        cache.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            for i, (c, d) in enumerate(zip(cands, ex.map(lambda c: describe(root, c["path"], cache), cands)), 1):
                c.update(d)
                if i % 100 == 0:
                    log(f"  deep {i}/{len(cands)}")
    cands = [c for c in cands if c.get("note") != "no default prim"]  # component layers cannot be referenced on their own
    for c in cands:
        c["pack"] = pack_of(c["path"])
    version = re.search(r"/(\d+\.\d+(?:\.\d+)?)/?$", root.rstrip("/"))
    meta = {"root": root, "version": version[1] if version else "local", "built": date.today().isoformat(), "n": len(cands),
            "deep": deep, "packs": list(packs),
            "license": "NVIDIA Omniverse License Agreement: assets are referenced by URL, not copied into dataset artifacts "
                       "(see docs/CATALOG.md)"}
    out = Path(out or LIBRARY / f"isaac_{meta['version']}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for c in sorted(cands, key=lambda c: c["path"]):
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    log(f"wrote {out} ({sum(c.get('deep', False) for c in cands)} deep)")
    return out


# ------------------------------------------------------------------------------------------------------- search
class CatalogIndex:
    """BM25 over asset tokens: name x3, labels x2, folder x1."""

    source = "omniverse"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        lines = self.path.read_text().splitlines()
        self.meta = json.loads(lines[0])["_meta"]
        self.entries = [json.loads(ln) for ln in lines[1:] if ln.strip()]
        self._prepare()
        self.by_path = {e["path"]: e for e in self.entries}
        self._docs = []
        for e in self.entries:
            toks = _with_compounds(_tokens(e["name"])) * 3 + [t for lab in e.get("labels", []) for t in _tokens(lab)] * 2 + _tokens(e["folder"])
            self._docs.append(Counter(toks))
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for d in self._docs for t in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _prepare(self):
        """Hook for subclasses: amend ``entries`` before they are tokenised."""

    def license(self, path: str) -> str:
        return license_of(path)

    def _qtokens(self, query: str) -> list[str]:
        toks = _tokens(query)
        extra = [s for t in _with_compounds(toks) for s in SYNONYMS.get(t, [])]
        return toks + [t for t in extra if t not in toks]

    def search(self, query: str, k: int = 8, tags: list[str] | None = None, max_size_m: float | None = None,
               scenes: bool = False) -> list[dict]:
        toks = self._qtokens(query) + [t for tag in (tags or []) for t in _tokens(tag)]
        k1, b = 1.4, 0.75
        scored = []
        for e, doc, ln in zip(self.entries, self._docs, self._len):
            if max_size_m and e.get("size_m") and max(e["size_m"]) > max_size_m:
                continue
            if not scenes and kind_of(e) == "scene":
                continue
            s = 0.0
            for t in toks:
                f = doc.get(t, 0)
                if f:
                    s += self._idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / self._avg))
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda x: (-x[0], x[1]["path"]))
        top = scored[0][0] if scored else 1.0
        return [{"path": e["path"], "name": e["name"], "folder": e["folder"], "score": round(s / top, 3), "raw_score": round(s, 3),
                 "size_m": e.get("size_m"), "physics": e.get("physics"), "labels": e.get("labels", []), "etag": e.get("etag", ""),
                 "license": self.license(e["path"]), "source": self.source} for s, e in scored[:k]]


_INDEX_CACHE: dict[str, CatalogIndex] = {}


def index_path() -> Path | None:
    if os.environ.get("WORLDC_CATALOG_INDEX"):
        return Path(os.environ["WORLDC_CATALOG_INDEX"])
    found = sorted(LIBRARY.glob("*.jsonl")) if LIBRARY.is_dir() else []
    return found[-1] if found else None


def load_index(path: Path | str | None = None) -> CatalogIndex | None:
    p = Path(path) if path else index_path()
    if p is None or not p.is_file():
        return None
    key = f"{p}:{p.stat().st_mtime_ns}"
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CatalogIndex(p)
    return _INDEX_CACHE[key]


INDEX_HINT = "Build one with `python -m worldc.assets.catalog_index index`, pin `path`, or use a registry asset."
MIN_SCORE = 0.35  # relative to the best possible hit for the query's own tokens


def search(query: str, k: int = 8, **kw) -> list[dict]:
    idx = load_index()
    return idx.search(query, k, **kw) if idx else []


def resolve_query(query: str, tags: list[str] | None = None, idx: CatalogIndex | None = None) -> dict | None:
    """Best catalog hit for ``query`` or None (no index, or nothing matches every query word well enough)."""
    idx = idx or load_index()
    if idx is None:
        return None
    hits = idx.search(query, 1, tags)
    if not hits:
        return None
    needed = set(_tokens(query))
    h = hits[0]
    have = set(_with_compounds(_tokens(h["name"])) + _tokens(h["folder"] + " " + " ".join(h["labels"])))
    q = _tokens(query)
    covered = needed & have
    for t in _with_compounds(q):  # a synonym present in the hit covers the word(s) it stands for ("hand truck" -> dolly)
        if set(SYNONYMS.get(t, [])) & have:
            covered |= {t} if t in needed else {w for w in q if w in t}
    return h if needed and len(covered) / len(needed) >= 2 / 3 else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="catalog_index")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--root", default=os.environ.get("WORLDC_ISAAC_ASSET_ROOT", DEFAULT_ROOT))
    a.add_argument("--out")
    a.add_argument("--packs", nargs="+", default=list(DEFAULT_PACKS))
    a.add_argument("--no-deep", action="store_true")
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--limit", type=int)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--index")
    ns = ap.parse_args(argv)
    if ns.cmd == "index":
        build_index(ns.root, ns.out, ns.packs, not ns.no_deep, ns.workers, ns.limit)
    else:
        idx = load_index(ns.index)
        if idx is None:
            raise SystemExit("no catalog index found; run `catalog_index index` or set WORLDC_CATALOG_INDEX")
        for h in idx.search(ns.query, ns.k):
            print(f"{h['score']:.2f}  {h['path']}  size={h['size_m']}  phys={h['physics']}")


if __name__ == "__main__":
    main()
