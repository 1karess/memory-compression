"""Paired bootstrap on the test split: RMSE differences between conditions, recovery rate R.

Pairs are matched by qid (every condition answered the same 2000 test queries). Writes results/test_stats.json.
R = (RMSE_mean - RMSE_C) / (RMSE_mean - RMSE_B): share of the full-memory gain over the no-memory
baseline that the 64-position cartridge keeps (A gives no numbers, so the mean baseline is the no-memory reference).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import ROOT, TARGET, load_splits  # noqa: E402

N_BOOT = 5000


def main():
    pq = pd.read_csv(ROOT / "outputs" / "main" / "final" / "final_test_per_query.csv")
    rows, subsets = load_splits()
    mem_mean = float(rows.set_index("id").loc[subsets["r0_n256"]["ids"], TARGET].mean())
    err = {c: g.sort_values("qid").set_index("qid")["pred_filled"] - g.sort_values("qid").set_index("qid")["y"]
           for c, g in pq[pq["cond"] != "A"].groupby("cond")}
    qids = err["B"].index
    y = pq[pq["cond"] == "B"].set_index("qid").loc[qids, "y"]
    err["mean"] = mem_mean - y
    E = {c: e.loc[qids].to_numpy() for c, e in err.items()}
    seeds = [c for c in E if c.startswith("C_s")]
    E["C_avg3"] = None  # the mean of the three seeds' RMSEs, bootstrapped jointly

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qids), size=(N_BOOT, len(qids)))

    def rmse_boot(e):
        return np.sqrt((e[idx] ** 2).mean(1))

    boots = {c: rmse_boot(e) for c, e in E.items() if e is not None}
    boots["C_avg3"] = np.mean([boots[s] for s in seeds], axis=0)
    point = {c: float(np.sqrt((e ** 2).mean())) for c, e in E.items() if e is not None}
    point["C_avg3"] = float(np.mean([point[s] for s in seeds]))

    def diff(a, b):
        d = boots[a] - boots[b]
        return {"diff": point[a] - point[b], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    out = {"n": len(qids), "rmse": point, "pairs": {}}
    for c in seeds + ["C_avg3"]:
        out["pairs"][f"{c} - B"] = diff(c, "B")
        out["pairs"][f"{c} - B_shuffle"] = diff(c, "B_shuffle")
        out["pairs"][f"{c} - mean"] = diff(c, "mean")
        out["pairs"][f"{c} - C_init"] = diff(c, "C_init")
    out["pairs"]["B - mean"] = diff("B", "mean")
    out["pairs"]["B - B_shuffle"] = diff("B", "B_shuffle")
    r = (boots["mean"] - boots["C_avg3"]) / (boots["mean"] - boots["B"])
    out["recovery_R"] = {"point": (point["mean"] - point["C_avg3"]) / (point["mean"] - point["B"]),
                         "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    (ROOT / "results" / "test_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
