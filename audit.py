"""Check a proposed full identity witness; never choose identities or optimize."""
from source import candidates, row_key, rows


def evaluate(pack, assignment):
    expected = {row_key(b, r) for b, r in rows(pack)}
    if set(assignment) != expected:
        raise ValueError("WITNESS_KEYS")
    used = set()
    versions = {}
    for batch, row in rows(pack):
        key = assignment[row_key(batch, row)]
        if key not in candidates(pack, batch, row) or (batch, key) in used:
            raise ValueError("WITNESS_IDENTITY")
        used.add((batch, key))
        item = {"rev": row["rev"], "deleted": row["deleted"], "amount": row["amount"]}
        version_key = (key, row["rev"])
        if version_key in versions and versions[version_key] != item:
            raise ValueError("WITNESS_REVISION_CONFLICT")
        versions[version_key] = item
    final = {}
    for (key, rev), item in versions.items():
        if key not in final or rev > final[key]["rev"]:
            final[key] = item
    region_of = {r["id"]: r["region"] for r in pack["registry"]}
    totals = {r["region"]: {"amount": 0, "count": 0} for r in pack["registry"]}
    for key, item in final.items():
        if not item["deleted"]:
            totals[region_of[key]]["amount"] += item["amount"]
            totals[region_of[key]]["count"] += 1
    return {"ledger": [{"id": key, **final[key]} for key in sorted(final)],
            "totals": dict(sorted(totals.items()))}


def check_result(pack, result):
    # This checks positive witnesses only. It does NOT certify completeness or optimality.
    if result["status"] != "ok" or result["materials"] != pack["materials"]:
        raise ValueError("RESULT_MATERIALS")
    evaluated = {wid: evaluate(pack, a) for wid, a in result["witnesses"].items()}
    expected = {row_key(b, r): (b, r) for b, r in rows(pack)}
    if set(result["row_support"]) != set(expected):
        raise ValueError("RESULT_ROWS")
    for key, support in result["row_support"].items():
        if not support:
            raise ValueError("EMPTY_SUPPORT")
        for identity, wid in support.items():
            if result["witnesses"][wid][key] != identity:
                raise ValueError("SUPPORT_WITNESS")
    regions = {r["region"] for r in pack["registry"]}
    if set(result["regions"]) != regions:
        raise ValueError("RESULT_REGIONS")
    for region, bound in result["regions"].items():
        if bound["observed"] != pack["archived_snapshot"]["totals"][region]["amount"]:
            raise ValueError("OBSERVED_AMOUNT")
        if bound["min"] > bound["max"]:
            raise ValueError("BOUND_ORDER")
        for side in ("min", "max"):
            actual = evaluated[bound[side + "_witness"]]["totals"][region]["amount"]
            if actual != bound[side]:
                raise ValueError("BOUND_WITNESS")
    return {"witness_count": len(evaluated), "row_count": len(expected)}
