"""NVIDIA SimReady assets from Hugging Face (``nvidia/simready-assets``): index, search, probe, fetch on demand.

    python -m worldc.assets.simready index [--revision main]      metadata only: one API call, no asset files
    python -m worldc.assets.simready search "toaster" [-k 8]
    python -m worldc.assets.simready info appliance_toaster_a01 [--probe]
    python -m worldc.assets.simready fetch appliance_toaster_a01 [--max-mb 1024]
    python -m worldc.assets.simready cached | clean [PACKAGE]

The dataset is ~1.7 k packages / ~260 GB, so nothing is stored up front and nothing is checked in.  ``index`` lists the
repo at one commit and writes a small JSONL (one line per package: root USD, bytes, file count, thumbnail) into the user
cache.  ``--probe`` downloads only the package's USD layers (no textures) to a temp dir, records bounds / physics /
semantic labels next to the index and deletes the layers.  A whole package is downloaded only when a spec uses it
(``{catalog: simready, query|path}``) or on ``fetch``; files are checked against the package BOM (sha256).

Everything is pinned to the commit the index was built from (``rev``), so a resolved spec replays the same files.
Unlike Omniverse catalog assets, the scene references the *cached local copy* (HF ``resolve`` URLs redirect to a CDN, so
relative payload / texture paths would not resolve against them).  License: CC-BY-4.0 (attribution required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from . import catalog_index as ci

REPO = "nvidia/simready-assets"
LICENSE = "cc-by-4.0"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldc" / "simready"
MAX_MB = float(os.environ.get("WORLDC_SIMREADY_MAX_MB", 1024))
INDEX_HINT = "Build it with `worldc simready index` (metadata only, no assets), pin `path`, or use a registry asset."
SKIP = ("/.thumbs/",)  # never needed to build or render


class SimReadyError(RuntimeError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.hint = code, hint


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def local_root() -> Path | None:
    """``WORLDC_SIMREADY_ROOT``: a local directory laid out like the repo (offline mirrors, tests)."""
    r = os.environ.get("WORLDC_SIMREADY_ROOT")
    return Path(r) if r else None


def file_url(rel: str, rev: str) -> str:
    return f"{endpoint()}/datasets/{REPO}/resolve/{rev}/{urllib.parse.quote(rel)}"


def _get(url: str, timeout=120):
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError as e:
        code = "asset.catalog_not_found" if e.code in (401, 403, 404) else "asset.catalog_unreachable"
        raise SimReadyError(code, f"{url} -> HTTP {e.code}", "Check the package name with `worldc simready search`.") from e
    except OSError as e:
        raise SimReadyError("asset.catalog_unreachable", f"Cannot reach {url}: {e}",
                            "Check network access or set WORLDC_SIMREADY_ROOT to a local mirror.") from e


# ------------------------------------------------------------------------------------------------------- index
def _listing(revision: str) -> tuple[str, list[tuple[str, int]]]:
    """(commit sha, [(path, size)]) of the repo at ``revision``, or of the local mirror."""
    root = local_root()
    if root:
        return "local", [(p.relative_to(root).as_posix(), p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()]
    with _get(f"{endpoint()}/api/datasets/{REPO}/revision/{urllib.parse.quote(revision)}?blobs=true", timeout=300) as r:
        d = json.load(r)
    return d["sha"], [(s["rfilename"], int(s.get("size") or 0)) for s in d["siblings"]]


def packages(files: list[tuple[str, int]]) -> list[dict]:
    """One record per package: ``<pkg>/simready_usd/<root>.usd`` plus totals (thumbnails excluded from ``bytes``)."""
    pk: dict[str, dict] = {}
    for rel, size in files:
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        p = pk.setdefault(parts[0], {"bytes": 0, "files": 0, "roots": [], "thumbs": []})
        if any(s in "/" + rel for s in SKIP):
            p["thumbs"].append(rel)
            continue
        p["bytes"] += size
        p["files"] += 1
        if len(parts) == 3 and parts[1] == "simready_usd" and rel.lower().endswith(ci.USD_EXT):
            p["roots"].append(rel)
    physx = {rel.split("/")[0] for rel, _ in files if "/payloads/PhysX/" in rel}  # SimReady's PhysX physics payload
    out = []
    for name, p in sorted(pk.items()):
        if not p["roots"]:
            continue  # not a SimReady package (or an incomplete upload)
        path = sorted(p["roots"])[0]
        stem = path.rsplit("/", 1)[1]
        thumb = f"{name}/simready_usd/.thumbs/256x256/{stem}.png"
        thumb = thumb if thumb in p["thumbs"] else None
        out.append({"path": path, "name": name, "folder": name, "package": name, "bytes": p["bytes"], "files": p["files"],
                    "physics": {"authored": name in physx}, **({"thumb": thumb} if thumb else {})})
    return out


def index_path() -> Path:
    return Path(os.environ.get("WORLDC_SIMREADY_INDEX") or CACHE / "index.jsonl")


def build_index(revision: str = "main", out: Path | str | None = None, log=print) -> Path:
    sha, files = _listing(revision)
    pkgs = packages(files)
    for e in pkgs:
        e["etag"] = sha  # the revision is per repo commit; it is also the download cache key
    meta = {"repo": REPO, "rev": sha, "revision": revision, "built": date.today().isoformat(), "n": len(pkgs), "license": LICENSE,
            "root": str(local_root()) if local_root() else f"{endpoint()}/datasets/{REPO}"}
    out = Path(out or index_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for e in pkgs:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    tmp.replace(out)
    log(f"wrote {out}: {len(pkgs)} packages at {sha[:12]} ({sum(e['bytes'] for e in pkgs) / 1e9:.0f} GB not downloaded)")
    return out


class SimReadyIndex(ci.CatalogIndex):
    source = "simready"

    def _prepare(self):
        for e in self.entries:  # probe results (bounds, labels) are kept next to the index, per revision
            pr = probe_file(e["package"], self.meta["rev"])
            if pr.is_file():
                e.update(json.loads(pr.read_text()))

    def license(self, path: str) -> str:
        return LICENSE

    def search(self, query, k=8, tags=None, max_size_m=None, scenes=False):
        rev = self.meta["rev"]
        out = []
        for h in super().search(query, k, tags, max_size_m, scenes):
            e = self.by_path[h["path"]]
            out.append({**h, "package": e["package"], "mb": round(e["bytes"] / 1e6, 1), "probed": "size_m" in e,
                        "cached": is_cached(e["package"], rev),
                        **({"thumb_url": file_url(e["thumb"], rev)} if e.get("thumb") and rev != "local" else {})})
        return out


_INDEX: dict[str, SimReadyIndex] = {}


def load_index(path: Path | str | None = None) -> SimReadyIndex | None:
    p = Path(path) if path else index_path()
    if not p.is_file():
        return None
    probes = probe_dir()
    key = f"{p}:{p.stat().st_mtime_ns}:{probes.stat().st_mtime_ns if probes.is_dir() else 0}"
    if key not in _INDEX:
        _INDEX[key] = SimReadyIndex(p)
    return _INDEX[key]


def search(query: str, k: int = 8, **kw) -> list[dict]:
    idx = load_index()
    return idx.search(query, k, **kw) if idx else []


def resolve_query(query: str, tags=None) -> dict | None:
    """Best hit whose name / labels literally contain >= 2/3 of the query words.  Stricter than the Omniverse rule: the
    synonym table ranks but never pins here ("forklift" must not pin a scissor lift table or a kitchen fork)."""
    idx = load_index()
    need = set(ci._tokens(query))
    for h in idx.search(query, 5, tags) if idx and need else []:
        have = set(ci._with_compounds(ci._tokens(h["name"])) + [t for lab in h["labels"] for t in ci._tokens(lab)])
        if len(need & have) / len(need) >= 2 / 3:
            return h
    return None


def entry(package: str) -> dict:
    idx = load_index()
    if idx is None:
        raise SimReadyError("asset.catalog_unavailable", "No SimReady index is installed.", INDEX_HINT)
    e = next((e for e in idx.entries if e["package"] == package or e["path"] == package), None)
    if e is None:
        near = ", ".join(h["package"] for h in idx.search(package.replace("_", " "), 3)) or "none"
        raise SimReadyError("asset.catalog_not_found", f"No SimReady package {package!r}.", f"Closest: {near}.")
    return e


# ------------------------------------------------------------------------------------------------------- probe
def probe_dir() -> Path:
    return index_path().parent / "probe"


def probe_file(package: str, rev: str) -> Path:
    return probe_dir() / rev[:12] / f"{package}.json"


def probe(package: str) -> dict:
    """Bounds (m, Z-up), authored physics and semantic labels from the USD layers only; the layers are deleted after."""
    e, rev = entry(package), load_index().meta["rev"]
    out = probe_file(e["package"], rev)
    if out.is_file():
        return json.loads(out.read_text())
    root = str(local_root()) if local_root() else f"{endpoint()}/datasets/{REPO}/resolve/{rev}"
    with tempfile.TemporaryDirectory(prefix="simready_probe_") as tmp:
        d = ci.describe(root, e["path"], Path(tmp))
    if not d.get("deep"):
        raise SimReadyError("asset.catalog_bad_usd", f"Cannot read {e['path']}: {d.get('note')}")
    d = {k: d[k] for k in ("size_m", "physics", "labels", "meshes")}
    d["physics"]["authored"] = e["physics"]["authored"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d))
    return d


# ------------------------------------------------------------------------------------------------------- fetch
def package_dir(package: str, rev: str) -> Path:
    return CACHE / "pkgs" / rev[:12] / package


def is_cached(package: str, rev: str) -> bool:
    return (package_dir(package, rev) / ".complete").is_file()


def _bom(package: str, rev: str) -> list[dict]:
    rel = f"{package}/.metadata/com.nvidia.simready.packaging.bom.json"
    if local_root():
        p = local_root() / rel
        return json.loads(p.read_text())["items"] if p.is_file() else [
            {"relative_path": f.relative_to(local_root() / package).as_posix()}
            for f in sorted((local_root() / package).rglob("*")) if f.is_file()]
    with _get(file_url(rel, rev)) as r:
        return json.load(r)["items"]


def fetch_package(package: str, max_mb: float | None = None, log=lambda *_: None) -> Path:
    """Download one package (minus thumbnails) into the cache at the index revision; returns its directory."""
    e, rev = entry(package), load_index().meta["rev"]
    d = package_dir(e["package"], rev)
    if is_cached(e["package"], rev):
        return d
    limit = MAX_MB if max_mb is None else max_mb
    if e["bytes"] > limit * 1e6:
        raise SimReadyError("asset.simready_too_large", f"{e['package']} is {e['bytes'] / 1e6:.0f} MB (limit {limit:.0f} MB).",
                            "Pick a smaller asset, or raise the limit (--max-mb / WORLDC_SIMREADY_MAX_MB).")
    items = [i for i in _bom(e["package"], rev) if not any(s in "/" + i["relative_path"] for s in SKIP)]
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"fetching {e['package']} ({e['bytes'] / 1e6:.0f} MB, {len(items)} files) at {rev[:12]}")
    for i in items:
        rel = i["relative_path"]
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local_root():
            shutil.copyfile(local_root() / e["package"] / rel, dst)
        else:
            with _get(file_url(f"{e['package']}/{rel}", rev)) as r, dst.open("wb") as f:
                shutil.copyfileobj(r, f)
        want = (i.get("hash") or {}).get("sha256")
        if want and hashlib.sha256(dst.read_bytes()).hexdigest() != want:
            shutil.rmtree(tmp, ignore_errors=True)
            raise SimReadyError("asset.simready_corrupt", f"{e['package']}/{rel}: sha256 does not match the package BOM.", "Retry the fetch.")
    meta = tmp / ".metadata"  # the BOM does not list the license / root files themselves
    for rel in (".metadata/LICENSE.txt",):
        if not (tmp / rel).exists():
            try:
                meta.mkdir(exist_ok=True)
                if local_root():
                    if (local_root() / e["package"] / rel).is_file():
                        shutil.copyfile(local_root() / e["package"] / rel, tmp / rel)
                else:
                    with _get(file_url(f"{e['package']}/{rel}", rev)) as r:
                        (tmp / rel).write_bytes(r.read())
            except SimReadyError:
                pass
    (tmp / ".complete").write_text(json.dumps({"rev": rev, "files": len(items)}))
    shutil.rmtree(d, ignore_errors=True)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(d)
    return d


def simready_asset(path: str, overrides: dict | None = None, rev: str | None = None):
    """MeshAsset for a SimReady root USD: fetch the package (once), import it, reference the cached original."""
    from .catalog import CatalogError, catalog_asset

    idx = load_index()
    if idx is None:
        raise SimReadyError("asset.catalog_unavailable", "No SimReady index is installed.", INDEX_HINT)
    if rev and rev != idx.meta["rev"]:
        raise SimReadyError("asset.simready_rev_mismatch", f"{path} was pinned at revision {rev[:12]}, the index is at {idx.meta['rev'][:12]}.",
                            f"Rebuild the index at that revision: `worldc simready index --revision {rev}`.")
    e = entry(path)
    pkg_dir = fetch_package(e["package"])
    ov = {"id": "simready_" + e["package"].lower(), **{k: v for k, v in (overrides or {}).items() if v not in (None, [], "")}}
    ov["tags"] = sorted(set(ov.get("tags", ())) | {"simready"})
    try:
        return catalog_asset(e["path"], ov, root=str(pkg_dir.parent), rev=idx.meta["rev"], source_tags=("catalog", "simready"))
    except CatalogError as err:
        raise SimReadyError(err.code, str(err), err.hint) from err


# ------------------------------------------------------------------------------------------------------- cli
def _cached() -> list[tuple[str, str, int]]:
    out = []
    for c in sorted((CACHE / "pkgs").glob("*/*/.complete")):
        d = c.parent
        out.append((d.parent.name, d.name, sum(f.stat().st_size for f in d.rglob("*") if f.is_file())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worldc simready", description="NVIDIA SimReady assets on Hugging Face: index, search, check, fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index", help="list the repo (metadata only) into the local index")
    a.add_argument("--revision", default="main", help="branch, tag or commit sha")
    s = sub.add_parser("search", help="rank packages for a query (builds the index on first use)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--max-mb", type=float, default=None, help="hide packages larger than this")
    s.add_argument("--json", action="store_true")
    i = sub.add_parser("info", help="details of one package")
    i.add_argument("package")
    i.add_argument("--probe", action="store_true", help="read bounds / physics / labels from its USD layers (no textures, nothing kept)")
    f = sub.add_parser("fetch", help="download one package into the cache")
    f.add_argument("package")
    f.add_argument("--max-mb", type=float, default=None)
    sub.add_parser("cached", help="list downloaded packages")
    c = sub.add_parser("clean", help="delete downloaded packages (all, or one)")
    c.add_argument("package", nargs="?")
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "index":
            build_index(ns.revision)
        elif ns.cmd == "search":
            if load_index() is None:
                build_index()
            hits = [h for h in search(ns.query, ns.k * 3) if ns.max_mb is None or h["mb"] <= ns.max_mb][:ns.k]
            if ns.json:
                print(json.dumps(hits, indent=1))
            for h in [] if ns.json else hits:
                size = "x".join(f"{v:.2f}" for v in h["size_m"]) + " m" if h.get("size_m") else "size ?"
                print(f"{h['score']:.2f}  {h['package']:42s} {h['mb']:7.1f} MB  {size:20s} {'[cached]' if h['cached'] else ''}")
            if not hits:
                print("no match")
        elif ns.cmd == "info":
            e = dict(entry(ns.package))
            rev = load_index().meta["rev"]
            if ns.probe:
                e.update(probe(e["package"]))
            e.update(rev=rev, license=LICENSE, cached=is_cached(e["package"], rev), page=f"{endpoint()}/datasets/{REPO}/tree/{rev}/{e['package']}",
                     use=f'"asset": {{"catalog": "simready", "path": "{e["path"]}"}}')
            if e.get("thumb") and rev != "local":
                e["thumb_url"] = file_url(e.pop("thumb"), rev)
            print(json.dumps(e, indent=1))
        elif ns.cmd == "fetch":
            print(fetch_package(ns.package, ns.max_mb, log=print))
        elif ns.cmd == "cached":
            rows = _cached()
            for rev, pkg, n in rows:
                print(f"{rev}  {pkg:42s} {n / 1e6:7.1f} MB")
            print(f"{len(rows)} package(s), {sum(r[2] for r in rows) / 1e6:.0f} MB in {CACHE / 'pkgs'}")
        elif ns.cmd == "clean":
            dirs = [CACHE / "pkgs" / r / p for r, p, _ in _cached() if ns.package in (None, p)]
            for d in dirs:
                shutil.rmtree(d)
            print(f"removed {len(dirs)} package(s)")
    except SimReadyError as e:
        raise SystemExit(f"{e.code}: {e} {e.hint}".strip())


if __name__ == "__main__":
    main()
