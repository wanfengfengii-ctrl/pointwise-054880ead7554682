"""Read and validate sealed evidence. No recovery search is performed here."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    blobs = {}
    for name, digest in manifest["sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("UNSAFE_PATH")
        raw = (root / path).read_bytes()
        if sha(raw) != digest:
            raise ValueError("MATERIAL_HASH " + name)
        blobs[name] = raw
    def text(name):
        return blobs[name].decode("utf-8")
    registry = list(csv.DictReader(io.StringIO(text("registry.csv"))))
    ids = [r["id"] for r in registry]
    if len(ids) != len(set(ids)) or not ids or any(
            not re.fullmatch(r"[0-9]+", x) for x in ids):
        raise ValueError("REGISTRY")
    if any(not r["region"] for r in registry):
        raise ValueError("REGION")
    routes = json.loads(text("routes.json"))
    batches = []
    seen = set()
    for desc in manifest["batches"]:
        bid = desc["id"]
        if bid in seen:
            raise ValueError("BATCH_DUPLICATE")
        seen.add(bid)
        rows = []
        rids = set()
        for raw in csv.DictReader(io.StringIO(text(desc["file"]))):
            rid, token, shard = raw["rid"], raw["token"], raw["shard"]
            if rid in rids or not rid or not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("ROW_ID_OR_TOKEN")
            rids.add(rid)
            if shard not in routes[bid] or raw["deleted"] not in ("0", "1"):
                raise ValueError("SHARD_OR_DELETE")
            rev, amount = int(raw["rev"]), int(raw["amount"])
            if rev <= 0:
                raise ValueError("REVISION")
            rows.append({"rid": rid, "token": token, "shard": shard,
                         "rev": rev, "deleted": raw["deleted"] == "1", "amount": amount})
        if len(rows) != desc["rows"]:
            raise ValueError("ROW_COUNT")
        batches.append({"id": bid, "rows": rows})
    for bid, shards in routes.items():
        if bid not in seen:
            raise ValueError("UNKNOWN_BATCH")
        for candidates in shards.values():
            if len(candidates) != len(set(candidates)) or any(x not in ids for x in candidates):
                raise ValueError("ROUTE_ID")
    return {"registry": registry, "routes": routes, "batches": batches,
            "archived_snapshot": json.loads(text("warehouse.json")),
            "archived_log": text("import.log").splitlines(),
            "materials": dict(sorted({**manifest["sha256"],
                "manifest.json": sha((root / "manifest.json").read_bytes())}.items()))}


def rows(pack):
    return [(b["id"], r) for b in pack["batches"] for r in b["rows"]]


def row_key(batch, row):
    return batch + "/" + row["rid"]


def candidates(pack, batch, row):
    return sorted(x for x in pack["routes"][batch][row["shard"]]
                  if str(int(x)) == row["token"])
