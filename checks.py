import json
import sys
from pathlib import Path
from importer import replay
from source import load
from audit import evaluate
from exporter import export


def normal():
    assert export([{"id": "07", "shard": "s", "rev": 1, "deleted": False, "amount": 0}]) == "rid,token,shard,rev,deleted,amount\n01,7,s,1,0,0\n"
    pack = {"registry": [{"id": "12", "region": "east"}, {"id": "34", "region": "west"}],
            "routes": {"a": {"s": ["12", "34"]}, "b": {"s": ["12", "34"]}},
            "batches": [
                {"id": "a", "rows": [
                    {"rid": "1", "token": "12", "shard": "s", "rev": 2, "deleted": False, "amount": 100},
                    {"rid": "2", "token": "34", "shard": "s", "rev": 1, "deleted": False, "amount": 0}]},
                {"id": "b", "rows": [
                    {"rid": "1", "token": "12", "shard": "s", "rev": 1, "deleted": False, "amount": 900},
                    {"rid": "2", "token": "34", "shard": "s", "rev": 3, "deleted": True, "amount": 0}]}]}
    actual, logs = replay(pack)
    expected = {"ledger": [
        {"id": "12", "rev": 2, "deleted": False, "amount": 100},
        {"id": "34", "rev": 3, "deleted": True, "amount": 0}],
        "totals": {"east": {"amount": 100, "count": 1}, "west": {"amount": 0, "count": 0}}}
    assert actual == expected
    assert logs == ["a: rows=2 accepted=2 ignored=0", "b: rows=2 accepted=1 ignored=1"]
    assert evaluate(pack, {"a/1": "12", "a/2": "34", "b/1": "12", "b/2": "34"}) == expected
    print("NORMAL_REGRESSION_OK")


def materials():
    pack = load("data")
    snap, logs = replay(pack)
    assert snap == pack["archived_snapshot"], (snap, pack["archived_snapshot"])
    assert logs == pack["archived_log"]
    assert len(pack["batches"]) == 4
    assert len({r["id"] for r in pack["registry"]}) == 10
    print("SEALED_REPLAY_OK")


def solver():
    import numpy as np
    from scipy.optimize import Bounds, milp
    res = milp(c=np.array([1.]), integrality=np.array([1]), bounds=Bounds([0.5], [2.]))
    assert res.success and round(res.x[0]) == 1
    print("SOLVER_RUNTIME_OK")


def demo():
    pack = load("data")
    snap, logs = replay(pack)
    print(json.dumps({"historical_import_log": logs, "historical_warehouse": snap},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "verify":
        normal()
        materials()
        solver()
    elif mode == "normal":
        normal()
    elif mode == "materials":
        materials()
    elif mode == "demo":
        demo()
    else:
        raise SystemExit("unknown mode")
