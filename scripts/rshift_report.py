"""Region counterfactual (jobs/13_region_shift.sh), CPU only.

Memory M'' = M with PE + SHIFT only where AT < median AT of M. On the test split, split queries by the same AT
threshold and report the mean prediction shift in each half for:
  teacher      B'' - B
  cartridge    C_rshift - C_main_s42
  regressors   OLS / 5-NN fit on B'' answers  -  the same regressor fit on B answers (same 1024 distillation inputs)
Ideal "follows the memory locally": about +SHIFT in the low-AT half and about 0 in the high-AT half.
Writes results/rshift_summary.json and prints a table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import FEATURES, ROOT, TARGET, load_splits, shifted_pe  # noqa: E402

C = ROOT / "outputs" / "ctrl"


def main():
    rows, subsets = load_splits()
    by_id = rows.set_index("id")
    mem_rows = by_id.loc[subsets["r0_n256"]["ids"]].reset_index()
    summ = json.loads((C / "teacher_rshift" / "summary.json").read_text())
    shift = summ["pe_shift"]
    _, thr = shifted_pe(mem_rows, shift, low_at_only=True)

    pq = pd.read_csv(C / "final_rshift" / "final_test_per_query.csv")
    qids = np.sort(pq.loc[pq["cond"] == "B", "qid"].unique())
    pred = {c: g.set_index("qid").loc[qids, "pred_filled"].to_numpy() for c, g in pq.groupby("cond") if c != "A"}
    Xt = by_id.loc[qids, FEATURES].to_numpy()
    yt = by_id.loc[qids, TARGET].to_numpy()
    low = by_id.loc[qids, "AT"].to_numpy() < thr

    def fit_predict(meta_csv):
        meta = pd.read_csv(meta_csv).dropna(subset=["teacher_pred"])
        X, yb = by_id.loc[meta["qid"], FEATURES].to_numpy(), meta["teacher_pred"].to_numpy()
        sc = StandardScaler().fit(X)
        return {"OLS": LinearRegression().fit(X, yb).predict(Xt),
                "5-NN": KNeighborsRegressor(5).fit(sc.transform(X), yb).predict(sc.transform(Xt))}

    reg_b = fit_predict(ROOT / "outputs" / "main" / "teacher" / "distill_meta.csv")
    reg_r = fit_predict(C / "teacher_rshift" / "distill_meta.csv")

    def halves(d):
        return {"low_AT_mean_shift": float(d[low].mean()), "high_AT_mean_shift": float(d[~low].mean()),
                "low_AT_sd": float(d[low].std()), "high_AT_sd": float(d[~low].std())}

    out = {"shift": shift, "AT_threshold": thr, "n_test_low": int(low.sum()), "n_test_high": int((~low).sum()),
           "teacher B'' - B": halves(pred["B_shift"] - pred["B"]),
           "C_rshift - C_main_s42": halves(pred["C_rshift"] - pred["C_main_s42"])}
    for name in reg_b:
        out[f"{name}(B'' answers) - {name}(B answers)"] = halves(reg_r[name] - reg_b[name])
    # how well each one tracks B'' itself
    out["rmse_vs_B''"] = {k: float(np.sqrt(np.mean((v - pred["B_shift"]) ** 2)))
                          for k, v in {"C_rshift": pred["C_rshift"], "OLS(B'')": reg_r["OLS"], "5-NN(B'')": reg_r["5-NN"]}.items()}
    out["test_rmse_vs_truth"] = {k: float(np.sqrt(np.mean((v - yt) ** 2))) for k, v in pred.items()}
    (ROOT / "results" / "rshift_summary.json").write_text(json.dumps(out, indent=2))
    print(f"shift={shift} AT threshold={thr:.2f}  low-AT test n={low.sum()}  high-AT n={(~low).sum()}")
    print(f"{'':40s} {'low AT':>8s} {'high AT':>8s}")
    for k, v in out.items():
        if isinstance(v, dict) and "low_AT_mean_shift" in v:
            print(f"{k:40s} {v['low_AT_mean_shift']:+8.2f} {v['high_AT_mean_shift']:+8.2f}")
    print(json.dumps({k: out[k] for k in ("rmse_vs_B''", "test_rmse_vs_truth")}, indent=2))


if __name__ == "__main__":
    main()
