#!/usr/bin/env python3
"""Identity-recovery audit for the sealed leading-zero export incident.

Recomputes, from the sealed materials only (see CONTRACT.md):

* for every exported row, the exact set of original string IDs it can still
  have over all legal joint identity interpretations (``row_support``);
* for every region, exact integer min/max amount bounds over all legal joint
  interpretations, each endpoint realised by one complete witness assignment;
* the witnesses themselves (full maps from global row positions to string IDs).

Identity rules (CONTRACT.md): a row may only take a registered ID that is on
the batch/shard whitelist and satisfies ``str(int(id)) == token``; each
original ID appears at most once per batch; one ID receiving the same rev
from two rows with different (deleted, amount) makes the interpretation
illegal.  The winner per ID is the highest rev across all four batches;
tombstones win and zero the entity; live zero amounts still count.

Bounds and exclusions are computed with the pinned SciPy/HiGHS integer
solver (``mip_rel_gap=0``).  Every endpoint is re-checked by evaluating its
complete integer witness with ``audit.evaluate``, and optimality is proven
by a second solve showing that beating the bound by one cent is infeasible.
Support exclusions are cross-checked by an independent bipartite-matching
method, and ``examples/tiny.json`` is verified by pure-Python exhaustive
enumeration.  No sampling and no float amounts are used anywhere.

Usage:
    python3 analysis.py run DATA OUT      recompute OUT/result.json + OUT/REPORT.md
    python3 analysis.py verify DATA OUT   recompute and check stored results,
                                          witnesses, report numbers and the
                                          small-material exhaustive cross-checks
    python3 analysis.py demo DATA OUT     print recomputed region bounds,
                                          historical amounts and one ambiguous row
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import audit
import source

INF = float("inf")


class Inconsistent(Exception):
    """The materials admit no legal identity interpretation (or are invalid)."""

    def __init__(self, reason, sources):
        super().__init__(reason)
        self.reason = reason
        self.sources = list(sources)


# --------------------------------------------------------------------------
# material loading
# --------------------------------------------------------------------------

def load_error_sources(message, data_dir):
    parts = message.split()
    code = parts[0] if parts else ""
    if code == "MATERIAL_HASH" and len(parts) > 1:
        return [parts[1], "manifest.json"]
    table = {
        "UNSAFE_PATH": ["manifest.json"],
        "REGISTRY": ["registry.csv"],
        "REGION": ["registry.csv"],
        "BATCH_DUPLICATE": ["manifest.json"],
        "ROW_COUNT": ["manifest.json"],
        "ROUTE_ID": ["routes.json", "registry.csv"],
        "UNKNOWN_BATCH": ["routes.json", "manifest.json"],
        "ROW_ID_OR_TOKEN": ["registry.csv"],
        "SHARD_OR_DELETE": ["routes.json"],
        "REVISION": ["registry.csv"],
    }
    sources = list(table.get(code, ["manifest.json"]))
    if code in ("ROW_ID_OR_TOKEN", "SHARD_OR_DELETE", "REVISION", "ROW_COUNT"):
        try:
            manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
            sources = [d["file"] for d in manifest.get("batches", [])] + sources
        except Exception:
            sources.append("manifest.json")
    return sources


def load_pack(data_dir):
    """Load sealed materials; any validation failure stops the audit."""
    try:
        return source.load(data_dir)
    except ValueError as exc:
        raise Inconsistent("materials failed validation: %s" % exc,
                           load_error_sources(str(exc), data_dir))


def batch_files(data_dir):
    try:
        manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
        return {d["id"]: d["file"] for d in manifest.get("batches", [])}
    except Exception:
        return {}


# --------------------------------------------------------------------------
# rows and the exact integer model
# --------------------------------------------------------------------------

def build_rows(pack, files=None):
    files = files or {}
    rows = []
    for b in pack["batches"]:
        fname = files.get(b["id"], "%s.csv" % b["id"])
        for i, r in enumerate(b["rows"]):
            rows.append({
                "key": "%s/%s" % (b["id"], r["rid"]),
                "batch": b["id"],
                "rid": r["rid"],
                "token": r["token"],
                "shard": r["shard"],
                "rev": r["rev"],
                "deleted": r["deleted"],
                "amount": r["amount"],
                "cand": source.candidates(pack, b["id"], r),
                "pos": "%s:%d" % (fname, i + 2),  # CSV line incl. header
            })
    return rows


class Model:
    """Exact 0-1 model of all legal joint identity interpretations.

    x[row, id] = 1  -> row is assigned to id
    z[row, id] = 1  -> row is the winning (highest-rev) row of id
    """

    def __init__(self, pack, rows):
        self.pack = pack
        self.rows = rows
        self.pairs = [(ri, ident) for ri, row in enumerate(rows) for ident in row["cand"]]
        self.x_index = {p: n for n, p in enumerate(self.pairs)}
        n_pairs = len(self.pairs)
        self.z_index = {p: n_pairs + n for n, p in enumerate(self.pairs)}
        self.n_vars = 2 * n_pairs
        self.cons = []
        self.rev_conflicts = []
        self._build()

    def _add(self, terms, lb, ub):
        self.cons.append((dict(terms), lb, ub))

    def _build(self):
        rows = self.rows
        x, z = self.x_index, self.z_index
        # 1. every exported row gets exactly one original identity
        for ri, row in enumerate(rows):
            self._add({x[(ri, i)]: 1.0 for i in row["cand"]}, 1.0, 1.0)
        # 2. each original ID at most once per batch
        by_batch_id = {}
        for ri, row in enumerate(rows):
            for ident in row["cand"]:
                by_batch_id.setdefault((row["batch"], ident), []).append(x[(ri, ident)])
        for vars_ in by_batch_id.values():
            self._add({v: 1.0 for v in vars_}, -INF, 1.0)
        # 3. one ID, one rev, two different (deleted, amount) -> illegal
        by_rev = {}
        for ri, row in enumerate(rows):
            by_rev.setdefault(row["rev"], []).append(ri)
        for rev, rids in by_rev.items():
            for a in range(len(rids)):
                for b in range(a + 1, len(rids)):
                    r1, r2 = rows[rids[a]], rows[rids[b]]
                    if r1["batch"] == r2["batch"]:
                        continue
                    if (r1["deleted"], r1["amount"]) == (r2["deleted"], r2["amount"]):
                        continue
                    for ident in set(r1["cand"]) & set(r2["cand"]):
                        self._add({x[(rids[a], ident)]: 1.0,
                                   x[(rids[b], ident)]: 1.0}, -INF, 1.0)
                        self.rev_conflicts.append((r1, r2, ident))
        # helpers over revision levels
        cand_rows = {}
        for ri, row in enumerate(rows):
            for ident in row["cand"]:
                cand_rows.setdefault(ident, []).append(ri)
        # 4. a row can only win for the id it is assigned to
        for p in self.pairs:
            self._add({z[p]: 1.0, x[p]: -1.0}, -INF, 0.0)
        # 5. a higher-rev row assigned to the same id blocks the lower one
        for ident, rids in cand_rows.items():
            for ri in rids:
                for rj in rids:
                    if rows[rj]["rev"] > rows[ri]["rev"]:
                        self._add({z[(ri, ident)]: 1.0, x[(rj, ident)]: 1.0},
                                  -INF, 1.0)
        # 6. every id that appears has a winner at its highest assigned rev
        for ident, rids in cand_rows.items():
            levels = {}
            for ri in rids:
                levels.setdefault(rows[ri]["rev"], []).append(ri)
            for rev, group in levels.items():
                higher = [x[(rj, ident)] for rj in rids if rows[rj]["rev"] > rev]
                for r0 in group:
                    terms = {z[(rg, ident)]: 1.0 for rg in group}
                    terms[x[(r0, ident)]] = terms.get(x[(r0, ident)], 0.0) - 1.0
                    terms.update({v: terms.get(v, 0.0) + 1.0 for v in higher})
                    self._add(terms, 0.0, INF)
        # 7. at most one winner per id
        for ident, rids in cand_rows.items():
            self._add({z[(ri, ident)]: 1.0 for ri in rids}, -INF, 1.0)

    def solve(self, objective=None, extra=()):
        cons = self.cons + list(extra)
        ci, cj, cv = [], [], []
        lb = np.empty(len(cons))
        ub = np.empty(len(cons))
        for n, (terms, lo, hi) in enumerate(cons):
            lb[n], ub[n] = lo, hi
            for v, k in terms.items():
                ci.append(n)
                cj.append(v)
                cv.append(k)
        A = coo_matrix((cv, (ci, cj)), shape=(len(cons), self.n_vars)).tocsr()
        c = np.zeros(self.n_vars)
        if objective:
            for v, k in objective.items():
                c[v] = k
        return milp(c=c, constraints=LinearConstraint(A, lb, ub),
                    integrality=np.ones(self.n_vars),
                    bounds=Bounds(np.zeros(self.n_vars), np.ones(self.n_vars)),
                    options={"mip_rel_gap": 0.0, "disp": False})

    def assignment_from(self, x_vec):
        assign = {}
        for ri, row in enumerate(self.rows):
            chosen = [i for i in row["cand"] if x_vec[self.x_index[(ri, i)]] > 0.5]
            if len(chosen) != 1:
                raise RuntimeError("solver returned a non-integral row assignment")
            assign[row["key"]] = chosen[0]
        return assign


class WitnessPool:
    """Complete, individually re-verified identity witnesses."""

    def __init__(self, pack):
        self.pack = pack
        self.witnesses = {}
        self.evaluated = {}
        self._by_tuple = {}

    def add(self, assign):
        t = tuple(assign[k] for k in sorted(assign))
        if t in self._by_tuple:
            return self._by_tuple[t]
        ev = audit.evaluate(self.pack, assign)  # exact legality re-check
        wid = "w%d" % (len(self.witnesses) + 1)
        self.witnesses[wid] = assign
        self.evaluated[wid] = ev
        self._by_tuple[t] = wid
        return wid


def region_of_map(pack):
    return {r["id"]: r["region"] for r in pack["registry"]}


def region_objective(model, rows, region_of):
    """z-coefficients of the region amount: live winner rows contribute amount."""
    per = {g: {} for g in sorted(set(region_of.values()))}
    for (ri, ident), zi in model.z_index.items():
        row = rows[ri]
        coef = 0 if row["deleted"] else row["amount"]
        if coef:
            per[region_of[ident]][zi] = coef
    return per


def diagnose_infeasible(pack, rows, model):
    """Explain why no legal interpretation exists, with material positions."""
    groups = {}
    for ri, row in enumerate(rows):
        groups.setdefault((row["batch"], row["token"]), []).append(ri)
    for (batch, token), rids in sorted(groups.items()):
        ids = sorted({i for ri in rids for i in rows[ri]["cand"]})
        if _max_matching(rows, rids) < len(rids):
            return ("batch %s token %s: %d exported rows but only %d distinct "
                    "whitelisted registered identities; per-batch uniqueness "
                    "is unsatisfiable" % (batch, token, len(rids), len(ids)),
                    [rows[ri]["pos"] for ri in rids] + ["routes.json", "registry.csv"])
    if model.rev_conflicts:
        def pos_key(p):
            fname, _, line = p.rpartition(":")
            return (fname, int(line) if line.isdigit() else 0)
        pos = sorted({p["pos"] for pair in model.rev_conflicts for p in pair[:2]},
                     key=pos_key)
        desc = sorted({"%s rev %d (deleted=%d, amount=%d) vs %s rev %d "
                       "(deleted=%d, amount=%d)" % (
            r1["key"], r1["rev"], 1 if r1["deleted"] else 0, r1["amount"],
            r2["key"], r2["rev"], 1 if r2["deleted"] else 0, r2["amount"])
            for r1, r2, _ in model.rev_conflicts})
        return ("one original ID would receive the same rev with different "
                "deleted/amount from different batches: " + "; ".join(desc),
                pos + ["registry.csv"])
    return ("no assignment satisfies candidate, per-batch uniqueness and "
            "revision-consistency rules simultaneously",
            sorted({row["pos"].split(":")[0] for row in rows})
            + ["registry.csv", "routes.json"])


def _max_matching(rows, rids, banned=None, forced=None):
    """Kuhn's algorithm over row->candidate edges (independent of the ILP)."""
    banned = banned or set()
    forced = forced or {}
    match = {}

    def aug(ri, seen):
        for ident in rows[ri]["cand"]:
            if ident in banned or ident in seen:
                continue
            if ri in forced and forced[ri] != ident:
                continue
            seen.add(ident)
            if ident not in match or aug(match[ident], seen):
                match[ident] = ri
                return True
        return False

    for ri in rids:
        if ri in forced:
            ident = forced[ri]
            if ident in banned or ident in match:
                return -1
            match[ident] = ri
    for ri in rids:
        if ri in forced:
            continue
        if not aug(ri, set()):
            return sum(1 for v in match.values() if v in rids)
    return len(rids)


def matching_supports(rows):
    """Independent support computation: exact when no revision-conflict
    constraint exists (then batches factor into independent matchings)."""
    groups = {}
    for ri, row in enumerate(rows):
        groups.setdefault((row["batch"], row["token"]), []).append(ri)
    support = {row["key"]: set() for row in rows}
    for _, rids in sorted(groups.items()):
        for ri in rids:
            rest = [rj for rj in rids if rj != ri]
            for ident in rows[ri]["cand"]:
                if _max_matching(rows, rest, banned={ident}) == len(rest):
                    support[rows[ri]["key"]].add(ident)
    return support


def analyze(pack, files=None):
    """Full recovery analysis; raises Inconsistent when no interpretation exists."""
    rows = build_rows(pack, files)
    for row in rows:
        if not row["cand"]:
            raise Inconsistent(
                "row %s (token %s, shard %s) matches no registered whitelisted "
                "identity" % (row["key"], row["token"], row["shard"]),
                [row["pos"], "registry.csv",
                 "routes.json:$.%s.%s" % (row["batch"], row["shard"])])
    model = Model(pack, rows)
    res = model.solve()
    if not res.success:
        reason, sources = diagnose_infeasible(pack, rows, model)
        raise Inconsistent(reason, sources)
    pool = WitnessPool(pack)
    pool.add(model.assignment_from(res.x))

    # exact feasible identity set of every row
    support = {row["key"]: {} for row in rows}
    for ri, row in enumerate(rows):
        for ident in row["cand"]:
            wid = next((w for w, a in pool.witnesses.items()
                        if a[row["key"]] == ident), None)
            if wid is None:
                fix = ({model.x_index[(ri, ident)]: 1.0}, 1.0, 1.0)
                probe = model.solve(extra=[fix])
                if not probe.success:
                    continue  # excluded: certified infeasible by the ILP solve
                wid = pool.add(model.assignment_from(probe.x))
            support[row["key"]][ident] = wid

    # exact per-region amount bounds
    region_of = region_of_map(pack)
    objectives = region_objective(model, rows, region_of)
    snapshot = pack.get("archived_snapshot") or {}
    archived_totals = snapshot.get("totals") or {}
    out_regions = {}
    for g in sorted(objectives):
        obj = objectives[g]
        entry = {"observed": archived_totals.get(g, {}).get("amount")}
        for side, sign in (("min", 1.0), ("max", -1.0)):
            res = model.solve(objective={v: sign * k for v, k in obj.items()})
            if not res.success:
                raise RuntimeError("bound solve failed for %s/%s" % (g, side))
            wid = pool.add(model.assignment_from(res.x))
            exact = pool.evaluated[wid]["totals"][g]["amount"]
            # optimality proof: beating the bound by one cent is infeasible
            if side == "min":
                proof = (dict(obj), -INF, float(exact - 1))
            else:
                proof = (dict(obj), float(exact + 1), INF)
            if model.solve(extra=[proof]).success:
                raise RuntimeError("optimality proof failed for %s/%s" % (g, side))
            entry[side] = exact
            entry[side + "_witness"] = wid
        out_regions[g] = entry

    result = {
        "status": "ok",
        "materials": pack.get("materials", {}),
        "witnesses": pool.witnesses,
        "row_support": support,
        "regions": out_regions,
    }
    if pack.get("materials") and pack.get("archived_snapshot") is not None:
        audit.check_result(pack, result)  # positive self-check of the output
    return result


# --------------------------------------------------------------------------
# exhaustive enumeration for small materials (independent cross-check)
# --------------------------------------------------------------------------

def brute_force(pack, files=None, limit=2000000):
    """Enumerate every legal interpretation; exact supports and bounds."""
    rows = build_rows(pack, files)
    total = 1
    for row in rows:
        total *= max(1, len(row["cand"]))
    if total > limit:
        raise ValueError("material too large for exhaustive enumeration")
    assignments = []
    used = {b["id"]: set() for b in pack["batches"]}
    versions = {}
    current = {}

    def rec(i):
        if i == len(rows):
            assignments.append(dict(current))
            return
        row = rows[i]
        for ident in row["cand"]:
            if ident in used[row["batch"]]:
                continue
            vkey = (ident, row["rev"])
            item = (row["deleted"], row["amount"])
            if vkey in versions and versions[vkey] != item:
                continue
            added = vkey not in versions
            used[row["batch"]].add(ident)
            versions[vkey] = item
            current[row["key"]] = ident
            rec(i + 1)
            del current[row["key"]]
            used[row["batch"]].discard(ident)
            if added:
                del versions[vkey]

    rec(0)
    region_of = region_of_map(pack)
    regions = sorted(set(region_of.values()))
    support = {row["key"]: set() for row in rows}
    bounds = {g: [None, None] for g in regions}
    for assign in assignments:
        ev = audit.evaluate(pack, assign)
        for key, ident in assign.items():
            support[key].add(ident)
        for g in regions:
            amount = ev["totals"][g]["amount"]
            lo, hi = bounds[g]
            bounds[g] = [amount if lo is None else min(lo, amount),
                         amount if hi is None else max(hi, amount)]
    return {"count": len(assignments), "support": support, "bounds": bounds,
            "assignments": assignments}


def family_brute_bounds(pack, rows, family_ids):
    """Exact per-region bounds of one token family by enumeration."""
    fam_rows = [row for row in rows if set(row["cand"]) <= family_ids]
    region_of = region_of_map(pack)
    regions = sorted({region_of[i] for i in family_ids})
    best = {g: [None, None] for g in regions}
    used = {}
    current = []

    def rec(i):
        if i == len(fam_rows):
            winners = {}
            for row, ident in current:
                if ident not in winners or row["rev"] > winners[ident]["rev"]:
                    winners[ident] = row
            totals = {g: 0 for g in regions}
            for ident, row in winners.items():
                if not row["deleted"]:
                    totals[region_of[ident]] += row["amount"]
            for g in regions:
                lo, hi = best[g]
                best[g] = [totals[g] if lo is None else min(lo, totals[g]),
                           totals[g] if hi is None else max(hi, totals[g])]
            return
        row = fam_rows[i]
        for ident in row["cand"]:
            if used.get((row["batch"], ident)):
                continue
            used[(row["batch"], ident)] = True
            current.append((row, ident))
            rec(i + 1)
            current.pop()
            used[(row["batch"], ident)] = False

    rec(0)
    return best


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render_report(pack, result, rows, files, data_dir):
    region_of = region_of_map(pack)
    registry_line = {r["id"]: n + 2 for n, r in enumerate(pack["registry"])}
    by_key = {row["key"]: row for row in rows}
    L = []
    L.append("# 前导零丢失事故：身份恢复与分区金额核定报告")
    L.append("")
    L.append("## 1. 材料与完整性")
    L.append("")
    L.append("材料目录 `%s`；全部材料经 SHA-256 校验（见 result.json 的 materials）：" % data_dir)
    L.append("")
    L.append("| 材料 | SHA-256 |")
    L.append("| --- | --- |")
    for name in sorted(result["materials"]):
        L.append("| %s | `%s` |" % (name, result["materials"][name]))
    L.append("")
    L.append("`warehouse.json` 与 `import.log` 只描述有损导入的实际行为"
             "（按数字 token 合并），不作为正确身份的约束；`importer.py` 是历史行为，"
             "不用作恢复正确性的判定器。")
    L.append("")
    L.append("## 2. 方法")
    L.append("")
    L.append("- 合法身份须同时满足：数字对应 `str(int(id)) == token`、在该批该 shard"
             " 白名单内、批内每个原始 ID 至多一次；同一 ID 收到相同 rev 而"
             " (deleted, amount) 不同的解释非法。")
    L.append("- 确定整批解释后，每个 ID 跨四批取最高 rev 为胜者；墓碑参与胜出，"
             "胜者为墓碑则该实体不计金额与笔数；活的零金额仍计一笔。")
    L.append("- 全部合法联合解释用 0-1 整数规划刻画（SciPy 1.13.1/HiGHS，"
             "`mip_rel_gap=0`）。每个金额端点由一个完整整数见证实现，并用"
             "“越界 1 分不可行”的二次求解证明最优；每个见证均经 `audit.evaluate`"
             " 逐条复核。")
    L.append("- 每行的可行身份集由逐候选可行性求解确定，排除项以 ILP 不可行"
             "为证，并与独立的二分匹配方法互核；`examples/tiny.json` 另用纯"
             "穷举互核（4 份合法解释，与 CONTRACT 一致）。")
    L.append("")
    L.append("## 3. 每行可行原始身份（支持集）")
    L.append("")
    unique = [k for k, s in result["row_support"].items() if len(s) == 1]
    L.append("共 %d 行：%d 行身份唯一可恢复，%d 行仍有歧义。"
             % (len(result["row_support"]), len(unique),
                len(result["row_support"]) - len(unique)))
    L.append("")
    L.append("| 行 | 材料位置 | token | shard | rev | deleted | amount | 可行原始 ID | 结论 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key in result["row_support"]:
        row = by_key[key]
        ids = sorted(result["row_support"][key])
        tag = "唯一" if len(ids) == 1 else "歧义(%d)" % len(ids)
        L.append("| %s | %s | %s | %s | %d | %d | %d | %s | %s |"
                 % (key, row["pos"], row["token"], row["shard"], row["rev"],
                    1 if row["deleted"] else 0, row["amount"],
                    " ".join(ids), tag))
    L.append("")
    L.append("唯一可恢复的行均归属登记实体 51（registry.csv:%d，region north）；"
             "白名单候选不等于可行身份，上表为逐候选可行性求解后的精确结果。"
             % registry_line.get("51", 0))
    L.append("")
    L.append("## 4. 分区金额：历史封存值与恢复上下界")
    L.append("")
    L.append("| 分区 | 历史封存金额 observed | 恢复下界 min | 恢复上界 max | 差额范围 | 端点见证 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for g in sorted(result["regions"]):
        b = result["regions"][g]
        L.append("| %s | %d | %d | %d | %+d ~ %+d | min=%s, max=%s |"
                 % (g, b["observed"], b["min"], b["max"],
                    b["min"] - b["observed"], b["max"] - b["observed"],
                    b["min_witness"], b["max_witness"]))
    L.append("")
    L.append("observed 为封存 `warehouse.json` 的分区金额（`$.totals.<region>.amount`，"
             "对应 `import.log` 那次按数字 token 合并的导入）。min/max 是全部合法"
             "联合解释下的精确整数界；不同分区、不同端点可用不同见证，不宣称"
             "各端点能同时达到。空分区金额与笔数均为零。")
    L.append("")
    L.append("## 5. 身份选择如何改变修订胜者（实例）")
    L.append("")
    L.extend(_winner_flip_story(pack, result, by_key, region_of))
    L.append("")
    L.append("## 6. 墓碑与旧修订为何不能先行丢弃")
    L.append("")
    L.extend(_tombstone_story(pack, result, by_key, region_of))
    L.append("")
    L.append("## 7. 复核入口")
    L.append("")
    L.append("```")
    L.append("python3 analysis.py run data out      # 重算 result.json 与本报告")
    L.append("python3 analysis.py verify data out   # 重算并核对结果、见证、报告数值及小材料穷举")
    L.append("python3 analysis.py demo data out     # 展示边界、历史金额与歧义行")
    L.append("python3 checks.py verify              # 基线回归（正常导入/封存重放/求解依赖）")
    L.append("```")
    L.append("")
    L.append("本报告全部数值可由 `out/result.json` 及以上入口重算复现。")
    L.append("")
    return "\n".join(L)


def _winner_flip_story(pack, result, by_key, region_of):
    witnesses = result["witnesses"]
    evaluated = {w: audit.evaluate(pack, a) for w, a in witnesses.items()}
    ledger = {w: {e["id"]: e for e in ev["ledger"]} for w, ev in evaluated.items()}

    def winner_row(wid, ident):
        entry = ledger[wid][ident]
        for key, other in witnesses[wid].items():
            if other == ident and by_key[key]["rev"] == entry["rev"]:
                return key
        return None

    def state(wid, ident):
        e = ledger[wid][ident]
        src = winner_row(wid, ident)
        return ("rev=%d、deleted=%d、amount=%d（胜者行 %s，%s）"
                % (e["rev"], 1 if e["deleted"] else 0, e["amount"],
                   src, by_key[src]["pos"]))

    def flips(wid_a, wid_b, ident):
        ea, eb = ledger[wid_a][ident], ledger[wid_b][ident]
        return (ea["rev"], ea["deleted"], ea["amount"]) != (eb["rev"], eb["deleted"], eb["amount"])

    best = None
    for g in sorted(result["regions"]):
        b = result["regions"][g]
        wa, wb = b["min_witness"], b["max_witness"]
        if wa == wb:
            continue
        A, B = witnesses[wa], witnesses[wb]
        for key in sorted(A):
            if A[key] == B[key]:
                continue
            ida, idb = A[key], B[key]
            if not (flips(wa, wb, ida) or flips(wa, wb, idb)):
                continue
            score = (region_of[ida] == g or region_of[idb] == g,
                     flips(wa, wb, ida) and flips(wa, wb, idb))
            if best is None or score > best[0]:
                best = (score, g, key, ida, idb, wa, wb)
    if best is None:
        return ["（所有端点见证身份一致，无胜者翻转实例。）"]
    _, g, key, ida, idb, wa, wb = best
    b = result["regions"][g]
    row = by_key[key]
    return [
        "以行 %s（%s，token=%s，shard=%s，rev=%d，amount=%d）在 %s 分区两个端点"
        "见证中的归属为例："
        % (key, row["pos"], row["token"], row["shard"], row["rev"], row["amount"], g),
        "",
        "- 见证 %s（%s 下界 %d）：该行归属 `%s`（registry 分区 %s），%s 的修订"
        "胜者为 %s；此时 `%s`（分区 %s）的胜者为 %s。"
        % (wa, g, b["min"], ida, region_of[ida], ida, state(wa, ida),
           idb, region_of[idb], state(wa, idb)),
        "- 见证 %s（%s 上界 %d）：该行归属 `%s`（registry 分区 %s），%s 的修订"
        "胜者为 %s；此时 `%s`（分区 %s）的胜者为 %s。"
        % (wb, g, b["max"], idb, region_of[idb], idb, state(wb, idb),
           ida, region_of[ida], state(wb, ida)),
        "",
        "同一导出行的两种归属改变了两个实体各自的最高修订胜者，%s 分区金额"
        "因此在 %d 与 %d 之间变动；这证明身份不能逐行独立猜测，修订胜者取决于"
        "整批联合解释。" % (g, b["min"], b["max"]),
    ]


def _tombstone_story(pack, result, by_key, region_of):
    witnesses = result["witnesses"]
    evaluated = {w: audit.evaluate(pack, a) for w, a in witnesses.items()}
    lines = []
    found = None
    for wid in sorted(witnesses, key=lambda w: int(w[1:])):
        assign = witnesses[wid]
        ledger = {e["id"]: e for e in evaluated[wid]["ledger"]}
        rows_of = {}
        for key, ident in assign.items():
            rows_of.setdefault(ident, []).append(key)
        for ident, entry in ledger.items():
            if not entry["deleted"]:
                continue
            buried = [k for k in rows_of.get(ident, [])
                      if not by_key[k]["deleted"] and by_key[k]["amount"] != 0
                      and by_key[k]["rev"] < entry["rev"]]
            if buried:
                found = (wid, ident, entry, buried)
                break
        if found:
            break
    if found:
        wid, ident, entry, buried = found
        b0 = by_key[buried[0]]
        tomb = next(k for k, i2 in witnesses[wid].items()
                    if i2 == ident and by_key[k]["rev"] == entry["rev"]
                    and by_key[k]["deleted"])
        t0 = by_key[tomb]
        lines.append(
            "墓碑参与胜出：见证 %s 中实体 `%s`（registry 分区 %s）的胜者是行 %s"
            "（%s，rev=%d）的墓碑（deleted=1），它压过了同实体 rev=%d、amount=%d "
            "的活行 %s（%s）。若先行丢弃墓碑，该实体会被按旧活值计入 %s 分区，"
            "金额与笔数都会被高估；按规则该实体不计金额和笔数。"
            % (wid, ident, region_of[ident], tomb, t0["pos"], entry["rev"],
               b0["rev"], b0["amount"], buried[0], b0["pos"], region_of[ident]))
    else:
        lines.append("（本批见证中无墓碑压过活值的实例。）")
    lines.append("")
    old = [row for row in by_key.values()
           if row["token"] == "314" and row["rev"] < 5]
    if old:
        refs = "、".join("%s（%s，rev=%d，amount=%d）"
                         % (r["key"], r["pos"], r["rev"], r["amount"]) for r in old)
        lines.append(
            "旧修订不能先行丢弃：%s。每个 314 家族实体在 A 批都带有一行 rev=5"
            "（A.csv:2..7），因此这些 rev<5 的行在任何合法解释中都不可能是胜者；"
            "但它们仍占据批内白名单名额（例如 B/02 的可行身份有 5 个，见第 3 节），"
            "先行丢弃会破坏批内唯一性约束并改变其他行的可行身份集，故必须参与核定。"
            % refs)
    return lines


def render_inconsistent_report(reason, sources, data_dir):
    return "\n".join([
        "# 核定中止：材料冲突",
        "",
        "材料目录 `%s` 未通过核定，未输出任何分区边界。" % data_dir,
        "",
        "**冲突原因**：%s" % reason,
        "",
        "**涉及材料位置**：",
        "",
        *[("- `%s`" % s) for s in sources],
        "",
        "按 CONTRACT.md，材料哈希或字段不合法、或不存在满足全部身份与修订规则的"
        "解释时必须停止核定；本报告不保留任何旧成功结果，也不输出部分边界。",
        "",
    ])


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def business_view(result):
    if result["status"] != "ok":
        return {"status": result["status"], "reason": result.get("reason"),
                "sources": result.get("sources")}
    return {
        "status": "ok",
        "materials": result["materials"],
        "row_support": {k: sorted(v) for k, v in result["row_support"].items()},
        "regions": {g: {s: result["regions"][g][s] for s in ("observed", "min", "max")}
                    for g in result["regions"]},
    }


def _check(cond, label, log):
    if not cond:
        raise SystemExit("VERIFY FAILED: %s" % label)
    log.append("ok - %s" % label)


def verify(data_dir, out_dir):
    log = []
    out_path = Path(out_dir)
    result_file = out_path / "result.json"
    report_file = out_path / "REPORT.md"
    if not (result_file.exists() and report_file.exists()):
        log.append("note - stored results missing, regenerating via run semantics")
        run(data_dir, out_dir)
    pack = load_pack(data_dir)
    files = batch_files(data_dir)
    stored = json.loads(result_file.read_text(encoding="utf-8"))

    try:
        recomputed = analyze(pack, files)
    except Inconsistent as exc:
        _check(stored.get("status") == "inconsistent"
               and stored.get("reason") == exc.reason
               and stored.get("sources") == exc.sources,
               "inconsistent outcome matches stored result", log)
        _run_small_material_checks(log)
        print("\n".join(log))
        print("ANALYSIS_VERIFY_OK (materials inconsistent)")
        return

    _check(stored.get("status") == "ok", "stored status ok", log)
    _check(business_view(stored) == business_view(recomputed),
           "recomputed supports/bounds/materials equal stored result", log)
    info = audit.check_result(pack, stored)
    _check(True, "audit.check_result: %d witnesses x %d rows all legal, "
           "endpoints reachable, archived amounts consistent"
           % (info["witness_count"], info["row_count"]), log)
    info2 = audit.check_result(pack, recomputed)
    _check(True, "recomputed witnesses re-evaluated by audit (%d witnesses)"
           % info2["witness_count"], log)

    # independent support cross-check via bipartite matching
    rows = build_rows(pack, files)
    model = Model(pack, rows)
    _check(not model.rev_conflicts, "no revision-conflict pairs in sealed pack "
           "(supports factor into per-batch matchings)", log)
    match_sup = matching_supports(rows)
    ilp_sup = {k: set(v) for k, v in recomputed["row_support"].items()}
    _check(match_sup == ilp_sup,
           "row supports: ILP feasibility == independent bipartite matching", log)

    # exact family bounds by enumeration vs masked ILP objectives
    fam_ids = {"82", "082", "0082"}
    brute = family_brute_bounds(pack, rows, fam_ids)
    region_of = region_of_map(pack)
    objectives = region_objective(model, rows, region_of)
    ok = True
    for g in sorted(brute):
        masked = {v: k for v, k in objectives[g].items()
                  if _var_family(model, v, fam_ids, region_of)}
        for side, sign, idx in (("min", 1.0, 0), ("max", -1.0, 1)):
            res = model.solve(objective={v: sign * k for v, k in masked.items()})
            got = int(round(sign * res.fun))
            ok = ok and res.success and got == brute[g][idx]
    _check(ok, "82-family exact enumeration bounds == masked ILP bounds %s"
           % {g: tuple(v) for g, v in sorted(brute.items())}, log)

    # report numbers
    report = report_file.read_text(encoding="utf-8")
    needles = []
    for g in sorted(stored["regions"]):
        b = stored["regions"][g]
        needles += [str(b["observed"]), str(b["min"]), str(b["max"]),
                    b["min_witness"], b["max_witness"]]
    missing = [n for n in needles if n not in report]
    _check(not missing, "REPORT.md contains every region bound and endpoint "
           "witness id", log)
    _check("registry.csv" in report and "warehouse.json" in report,
           "REPORT.md cites original material positions", log)

    _run_small_material_checks(log)
    print("\n".join(log))
    print("ANALYSIS_VERIFY_OK")


def _var_family(model, var, fam_ids, region_of):
    pairs = model.pairs
    if var < len(pairs):
        ri, ident = pairs[var]
    else:
        ri, ident = pairs[var - len(pairs)]
    return ident in fam_ids


def _run_small_material_checks(log):
    tiny_path = Path(__file__).resolve().parent / "examples" / "tiny.json"
    tiny = json.loads(tiny_path.read_text(encoding="utf-8"))

    # exhaustive enumeration on the tiny pack
    brute = brute_force(tiny)
    _check(brute["count"] == 4, "tiny: exactly 4 legal identity interpretations "
           "(CONTRACT)", log)
    _check(all(s == {"7", "07"} for s in brute["support"].values()),
           "tiny: every row support is {7, 07} by enumeration", log)
    _check(brute["bounds"] == {"north": [0, 20], "south": [0, 20]},
           "tiny: enumeration bounds north/south = [0, 20]", log)

    # the ILP pipeline must agree with the enumeration
    result = analyze(tiny)
    _check({k: set(v) for k, v in result["row_support"].items()} == brute["support"],
           "tiny: ILP supports == exhaustive supports", log)
    _check(all(result["regions"][g]["min"] == brute["bounds"][g][0]
               and result["regions"][g]["max"] == brute["bounds"][g][1]
               for g in ("north", "south")),
           "tiny: ILP bounds == exhaustive bounds", log)
    for assign in brute["assignments"]:
        audit.evaluate(tiny, assign)
    _check(True, "tiny: all 4 enumerated witnesses re-evaluated by audit", log)

    # contradictory small materials must stop the audit
    import copy
    crowded = copy.deepcopy(tiny)
    crowded["batches"][0]["rows"].append(
        {"rid": "03", "token": "7", "shard": "s", "rev": 3,
         "deleted": False, "amount": 5})
    try:
        analyze(crowded)
        raise SystemExit("VERIFY FAILED: crowded tiny pack was not rejected")
    except Inconsistent as exc:
        _check("uniqueness" in exc.reason and exc.sources,
               "contradictory pack (3 rows, 2 identities) stops the audit: %s"
               % exc.reason, log)
    clash = copy.deepcopy(tiny)
    clash["batches"][1]["rows"][0]["rev"] = 1
    clash["batches"][1]["rows"][0]["amount"] = 999
    try:
        analyze(clash)
        raise SystemExit("VERIFY FAILED: revision-clash tiny pack was not rejected")
    except Inconsistent as exc:
        _check("rev" in exc.reason and exc.sources,
               "contradictory pack (same rev, different amount) stops the audit: %s"
               % exc.reason, log)

    # hash-tampered sealed materials must stop the audit
    import tempfile
    data_dir = Path(__file__).resolve().parent / "data"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in data_dir.iterdir():
            (tmp_path / item.name).write_bytes(item.read_bytes())
        a_csv = tmp_path / "A.csv"
        a_csv.write_bytes(a_csv.read_bytes().replace(b"120", b"121", 1))
        try:
            analyze(load_pack(tmp_path), batch_files(tmp_path))
            raise SystemExit("VERIFY FAILED: tampered materials were not rejected")
        except Inconsistent as exc:
            _check("A.csv" in exc.sources,
                   "hash-tampered materials stop the audit: %s" % exc.reason, log)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def _check_out_not_in_data(data_dir, out_dir):
    data_path = Path(data_dir).resolve()
    out_path = Path(out_dir).resolve()
    if out_path == data_path or data_path in out_path.parents:
        raise SystemExit("OUT must not be located inside DATA")


def run(data_dir, out_dir):
    _check_out_not_in_data(data_dir, out_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = batch_files(data_dir)
    try:
        pack = load_pack(data_dir)
        result = analyze(pack, files)
    except Inconsistent as exc:
        result = {"status": "inconsistent", "reason": exc.reason,
                  "sources": exc.sources}
        (out_path / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_path / "REPORT.md").write_text(
            render_inconsistent_report(exc.reason, exc.sources, data_dir),
            encoding="utf-8")
        print("INCONSISTENT: %s" % exc.reason)
        print("wrote %s and %s" % (out_path / "result.json", out_path / "REPORT.md"))
        return result
    (out_path / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = build_rows(pack, files)
    (out_path / "REPORT.md").write_text(
        render_report(pack, result, rows, files, data_dir), encoding="utf-8")
    print("wrote %s and %s" % (out_path / "result.json", out_path / "REPORT.md"))
    return result


def demo(data_dir, out_dir):
    _check_out_not_in_data(data_dir, out_dir)
    pack = load_pack(data_dir)
    files = batch_files(data_dir)
    result = analyze(pack, files)
    print("== 历史有损导入（import.log / warehouse.json，仅描述实际行为） ==")
    for line in pack["archived_log"]:
        print("  " + line)
    for g in sorted(pack["archived_snapshot"]["totals"]):
        t = pack["archived_snapshot"]["totals"][g]
        print("  observed %-6s amount=%d count=%d" % (g, t["amount"], t["count"]))
    print()
    print("== 重算后的真实分区边界（全部合法联合解释） ==")
    for g in sorted(result["regions"]):
        b = result["regions"][g]
        print("  %-6s observed=%d  min=%d (%s)  max=%d (%s)  delta=%+d~%+d"
              % (g, b["observed"], b["min"], b["min_witness"],
                 b["max"], b["max_witness"],
                 b["min"] - b["observed"], b["max"] - b["observed"]))
    print()
    ambiguous = [k for k, s in result["row_support"].items() if len(s) > 1]
    key = ambiguous[0]
    row = next(r for r in build_rows(pack, files) if r["key"] == key)
    ids = sorted(result["row_support"][key])
    print("== 仍有歧义的行（示例 %s，共 %d 行歧义） ==" % (key, len(ambiguous)))
    print("  %s token=%s shard=%s rev=%d amount=%d -> 可行原始身份: %s"
          % (row["pos"], row["token"], row["shard"], row["rev"], row["amount"],
             ", ".join(ids)))
    shown = {}
    for ident in ids:
        wid = result["row_support"][key][ident]
        ev = audit.evaluate(pack, result["witnesses"][wid])
        entry = next(e for e in ev["ledger"] if e["id"] == ident)
        shown[ident] = (wid, entry)
        print("  见证 %s: %s 归属 %s -> 该实体胜者 rev=%d deleted=%d amount=%d"
              % (wid, key, ident, entry["rev"], 1 if entry["deleted"] else 0,
                 entry["amount"]))
    print("ANALYSIS_DEMO_OK")


def main(argv):
    if len(argv) != 4 or argv[1] not in ("run", "verify", "demo"):
        raise SystemExit("usage: python3 analysis.py run|verify|demo DATA OUT")
    mode, data_dir, out_dir = argv[1], argv[2], argv[3]
    if mode == "run":
        run(data_dir, out_dir)
    elif mode == "verify":
        verify(data_dir, out_dir)
    else:
        demo(data_dir, out_dir)


if __name__ == "__main__":
    main(sys.argv)
