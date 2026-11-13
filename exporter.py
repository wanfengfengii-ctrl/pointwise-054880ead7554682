"""Archived numeric export adapter; raw source rows are no longer available."""
import csv
import io


def export(records):
    records = list(records)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("DUPLICATE_SOURCE_ID")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["rid", "token", "shard", "rev", "deleted", "amount"])
    for i, record in enumerate(records, 1):
        writer.writerow([str(i).zfill(2), str(int(record["id"])), record["shard"],
                         record["rev"], int(record["deleted"]), record["amount"]])
    return out.getvalue()
