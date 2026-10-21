"""Summarize the information-source controls (jobs/10_controls.sh), CPU only.

For every control cartridge: test RMSE vs truth, agreement with B (and with B-shuffle for shufT), dev selection;
next to it, OLS / 1-NN / 5-NN fit on exactly that run's distillation inputs and teacher answers (no memory),
so each C is compared with "a plain regressor given the same supervision". Writes results/ctrl_summary.json
and results/ctrl_summary.md.
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
from ccpp_common import FEATURES, ROOT, load_splits  # noqa: E402

C = ROOT / "outputs" / "ctrl"
MAIN_TD = ROOT / "outputs" / "main" / "teacher"
# cond in the test table -> (run dir, teacher meta csv the run was trained on, number of rows used, reference cond)
RUNS = {
    "C_main_s42": (ROOT / "outputs" / "main" / "runs" / "lr2e-2_s42", MAIN_TD / "distill_meta.csv", 1024, "B"),
    "C_shufT": (C / "runs" / "shufT", C / "teacher_shuf" / "distill_meta.csv", 1024, "B_shuffle"),
    "C_randinit": (C / "runs" / "randinit", MAIN_TD / "distill_meta.csv", 1024, "B"),
    "C_n256": (C / "runs" / "n256", MAIN_TD / "distill_meta.csv", 256, "B"),
    "C_n64": (C / "runs" / "n64", MAIN_TD / "distill_meta.csv", 64, "B"),
    "C_synth": (C / "runs" / "synth", C / "teacher_synth" / "synth_meta.csv", 1024, "B"),
    "C_shift": (C / "runs" / "shift", C / "teacher_shift" / "distill_meta.csv", 1024, "B_shift"),
    "C_shift_initM": (C / "runs" / "shift_initM", C / "teacher_shift" / "distill_meta.csv", 1024, "B_shift"),
}


def load_test_tables():
    """final (jobs/10) and final_shift (jobs/12) answer the same test queries; B and C_main_s42 are in both,
    so each condition is taken from the first table that has it."""
    parts = []
    for d in ("final", "final_shift"):
        f = C / d / "final_test_per_query.csv"
        if f.exists():
            df = pd.read_csv(f)
            seen = {c for p in parts for c in p["cond"].unique()}
            parts.append(df[~df["cond"].isin(seen)])
    return pd.concat(parts, ignore_index=True)


def agreement(p, ref):
    return {"corr": float(np.corrcoef(p, ref)[0, 1]), "mean_abs_diff": float(np.mean(np.abs(p - ref)))}


def main():
    rows, _ = load_splits()
    by_id = rows.set_index("id")
    pq = load_test_tables()
    qids = np.sort(pq.loc[pq["cond"] == "B", "qid"].unique())
    yt = by_id.loc[qids, "PE"].to_numpy()
    Xt = by_id.loc[qids, FEATURES].to_numpy()
    pred = {c: g.set_index("qid").loc[qids, "pred_filled"].to_numpy() for c, g in pq.groupby("cond") if c != "A"}
    parse = {c: float(g["parse_ok"].mean()) for c, g in pq.groupby("cond") if c != "A"}
    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))  # noqa: E731

    out = {"references": {c: {"test_rmse": rmse(pred[c]), "parse_rate": parse[c]}
                          for c in ("B", "B_shuffle", "B_shift", "C_randinit_init") if c in pred}}
    ols_preds = {}
    for cond, (run, meta_csv, n, ref) in RUNS.items():
        if cond not in pred:
            continue
        sel = json.loads((run / "dev_selection.json").read_text())
        meta = pd.read_csv(meta_csv).head(n).dropna(subset=["teacher_pred"])
        X = meta[FEATURES].to_numpy() if set(FEATURES) <= set(meta.columns) else by_id.loc[meta["qid"], FEATURES].to_numpy()
        yb = meta["teacher_pred"].to_numpy()
        sc = StandardScaler().fit(X)
        regs = {}
        for name, m in {"OLS": LinearRegression(), "1-NN": KNeighborsRegressor(1),
                        "5-NN": KNeighborsRegressor(min(5, len(X)))}.items():
            Xf, Xq = (sc.transform(X), sc.transform(Xt)) if "NN" in name else (X, Xt)
            p = m.fit(Xf, yb).predict(Xq)
            regs[name] = {"test_rmse": rmse(p), f"vs_{ref}": agreement(p, pred[ref])}
            if name == "OLS":
                ols_preds[cond] = p
        out[cond] = {
            "n_distill": len(meta), "reference_teacher": ref,
            "test_rmse": rmse(pred[cond]), "parse_rate": parse[cond],
            f"vs_{ref}": agreement(pred[cond], pred[ref]),
            "vs_B": agreement(pred[cond], pred["B"]),
            "best_dev_kl": sel["best_dev_kl"], "init_dev_kl": sel["init_dev_kl"], "best_updates": sel["best_updates"],
            "regressors_on_same_teacher_answers": regs,
        }
    # counterfactual memory: mean test-prediction shift (X under M') - (X under M); the memory shift is SHIFT for all rows
    if "B_shift" in pred:
        shift = json.loads((C / "teacher_shift" / "summary.json").read_text())["pe_shift"]
        d = lambda a, b: {"mean": float(np.mean(a - b)), "sd": float(np.std(a - b))}  # noqa: E731
        out["shift_test"] = {"memory_pe_shift": shift, "B_shift - B": d(pred["B_shift"], pred["B"])}
        for c in ("C_shift", "C_shift_initM"):
            if c in pred and "C_main_s42" in pred:
                out["shift_test"][f"{c} - C_main_s42"] = d(pred[c], pred["C_main_s42"])
        if "C_shift" in ols_preds and "C_main_s42" in ols_preds:
            out["shift_test"]["OLS(B' answers) - OLS(B answers)"] = d(ols_preds["C_shift"], ols_preds["C_main_s42"])
    (ROOT / "results" / "ctrl_summary.json").write_text(json.dumps(out, indent=2))

    lines = ["| C | 蒸馏条数 | 参照老师 | C 测试 RMSE | C 与老师相关 | C 与老师数字差 | OLS RMSE | 5-NN RMSE | 5-NN 与老师相关 | 5-NN 与老师数字差 | 开发 KL |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in RUNS:
        if cond not in out:
            continue
        r = out[cond]
        ref = r["reference_teacher"]
        g = r["regressors_on_same_teacher_answers"]
        lines.append(f"| {cond} | {r['n_distill']} | {ref} | {r['test_rmse']:.2f} | {r[f'vs_{ref}']['corr']:.3f} | "
                     f"{r[f'vs_{ref}']['mean_abs_diff']:.2f} | {g['OLS']['test_rmse']:.2f} | {g['5-NN']['test_rmse']:.2f} | "
                     f"{g['5-NN'][f'vs_{ref}']['corr']:.3f} | {g['5-NN'][f'vs_{ref}']['mean_abs_diff']:.2f} | {r['best_dev_kl']:.4f} |")
    lines.append("")
    lines += [f"- {c}: test RMSE {v['test_rmse']:.2f}, parse rate {v['parse_rate']:.3f}" for c, v in out["references"].items()]
    if "shift_test" in out:
        lines += ["", f"反事实记忆（M 中所有 PE + {out['shift_test']['memory_pe_shift']:g}）：测试集平均预测平移", ""]
        lines += [f"- {k}: {v['mean']:+.2f} (sd {v['sd']:.2f})" for k, v in out["shift_test"].items() if isinstance(v, dict)]
    (ROOT / "results" / "ctrl_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
