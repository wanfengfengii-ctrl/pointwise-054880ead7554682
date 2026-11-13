"""Historical importer: reproduce the archived run, not an identity restorer."""
import sqlite3


def connect():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE ledger(id TEXT PRIMARY KEY, rev INTEGER, deleted INTEGER, amount INTEGER)")
    return db


def apply(db, batch, rows):
    accepted = ignored = 0
    with db:
        for row in rows:
            key = str(int(row["token"]))
            old = db.execute("SELECT rev FROM ledger WHERE id=?", (key,)).fetchone()
            if old is not None and old[0] >= row["rev"]:
                ignored += 1
                continue
            db.execute("INSERT OR REPLACE INTO ledger VALUES(?,?,?,?)",
                       (key, row["rev"], int(row["deleted"]), row["amount"]))
            accepted += 1
    return f"{batch}: rows={len(rows)} accepted={accepted} ignored={ignored}"


def snapshot(db, registry):
    regions = {r["id"]: r["region"] for r in registry}
    totals = {r["region"]: {"amount": 0, "count": 0} for r in registry}
    ledger = []
    for key, rev, deleted, amount in db.execute("SELECT * FROM ledger ORDER BY id"):
        ledger.append({"id": key, "rev": rev, "deleted": bool(deleted), "amount": amount})
        if not deleted:
            totals[regions[key]]["amount"] += amount
            totals[regions[key]]["count"] += 1
    return {"ledger": ledger, "totals": dict(sorted(totals.items()))}


def replay(pack):
    db = connect()
    try:
        logs = [apply(db, b["id"], b["rows"]) for b in pack["batches"]]
        return snapshot(db, pack["registry"]), logs
    finally:
        db.close()
