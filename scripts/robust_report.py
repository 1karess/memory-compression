"""Stage-0 robustness summary (jobs/15, 16, 17), CPU only. Missing pieces are skipped, so it can run on partial results.

  1. seeds   : shufT / n64 / synth at seeds 42, 43, 44 (test RMSE, agreement with their teacher)
  2. rshift  : region counterfactual at seeds 42, 43, 44 (low/high-AT mean shift vs each seed's own main C)
  3. memories: r0 / r1 / r2_n256 (B, B-shuffle, C x 3 seeds, recovery R, regressors, region counterfactual at seed 42)
  4. k sweep : k = 16 / 64 / 256 x 3 seeds on r0
Writes results/robust_summary.json and results/robust_summary.md.
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

OUT = ROOT / "outputs"
ROWS, SUBSETS = load_splits()
BY_ID = ROWS.set_index("id")
TEST_QIDS = np.sort(ROWS.loc[ROWS["split"] == "test", "id"].to_numpy())
YT = BY_ID.loc[TEST_QIDS, TARGET].to_numpy()
XT = BY_ID.loc[TEST_QIDS, FEATURES].to_numpy()


def preds(path):
    """cond -> test predictions (mean-filled), ordered by TEST_QIDS; {} if the table does not exist."""
    f = Path(path) / "final_test_per_query.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return {c: g.set_index("qid").loc[TEST_QIDS, "pred_filled"].to_numpy() for c, g in df.groupby("cond") if c != "A"}


def rmse(p, ref=YT):
    return float(np.sqrt(np.mean((p - ref) ** 2)))


def agree(p, ref):
    return {"corr": float(np.corrcoef(p, ref)[0, 1]), "abs_diff": float(np.mean(np.abs(p - ref)))}


def msd(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v), "values": v.round(4).tolist()}


def regressors(meta_csv, n=None):
    """OLS / 5-NN fit on a teacher-data file's (x, teacher answer) pairs; test predictions."""
    meta = pd.read_csv(meta_csv)
    if n:
        meta = meta.head(n)
    meta = meta.dropna(subset=["teacher_pred"])
    X = meta[FEATURES].to_numpy() if set(FEATURES) <= set(meta.columns) else BY_ID.loc[meta["qid"], FEATURES].to_numpy()
    y = meta["teacher_pred"].to_numpy()
    sc = StandardScaler().fit(X)
    return {"OLS": LinearRegression().fit(X, y).predict(XT),
            "5-NN": KNeighborsRegressor(min(5, len(X))).fit(sc.transform(X), y).predict(sc.transform(XT))}


def region(p_b2, p_b, p_c2, p_c, thr, regs_b2=None, regs_b=None):
    low = BY_ID.loc[TEST_QIDS, "AT"].to_numpy() < thr
    half = lambda d: {"low": float(d[low].mean()), "high": float(d[~low].mean())}  # noqa: E731
    r = {"teacher": half(p_b2 - p_b), "C": half(p_c2 - p_c), "C_rmse_vs_teacher2": rmse(p_c2, p_b2)}
    if regs_b2 is not None:
        for k in regs_b2:
            r[k] = half(regs_b2[k] - regs_b[k])
            r[f"{k}_rmse_vs_teacher2"] = rmse(regs_b2[k], p_b2)
    return r


def main():
    out = {}
    ctrl, main_ = OUT / "ctrl", OUT / "main"
    p_ctrl, p_seeds, p_main = preds(ctrl / "final"), preds(ctrl / "final_seeds"), preds(main_ / "final")

    # 1. control seeds
    sec = {}
    for name, ref in [("shufT", "B_shuffle"), ("n64", "B"), ("synth", "B")]:
        runs = [p_ctrl.get(f"C_{name}")] + [p_seeds.get(f"C_{name}_s{s}") for s in (43, 44)]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        refp = p_ctrl[ref]
        sec[name] = {"teacher": ref, "test_rmse": msd([rmse(r) for r in runs]),
                     "corr_with_teacher": msd([agree(r, refp)["corr"] for r in runs]),
                     "abs_diff_with_teacher": msd([agree(r, refp)["abs_diff"] for r in runs])}
    out["control_seeds"] = sec

    # 2. region counterfactual seeds (r0)
    thr0 = shifted_pe(BY_ID.loc[SUBSETS["r0_n256"]["ids"]].reset_index(), 10, True)[1]
    p_r42, p_rs = preds(ctrl / "final_rshift"), preds(ctrl / "final_rshift_seeds")
    per_seed = []
    if p_r42:
        per_seed.append(region(p_r42["B_shift"], p_r42["B"], p_r42["C_rshift"], p_r42["C_main_s42"], thr0))
    for s in (43, 44):
        if f"C_rshift_s{s}" in p_rs:
            per_seed.append(region(p_rs["B_shift"], p_rs["B"], p_rs[f"C_rshift_s{s}"], p_rs[f"C_main_s{s}"], thr0))
    if per_seed:
        out["rshift_seeds"] = {"teacher_low": per_seed[0]["teacher"]["low"], "teacher_high": per_seed[0]["teacher"]["high"],
                               "C_low": msd([r["C"]["low"] for r in per_seed]), "C_high": msd([r["C"]["high"] for r in per_seed]),
                               "C_rmse_vs_teacher2": msd([r["C_rmse_vs_teacher2"] for r in per_seed])}
        regs_b2 = regressors(ctrl / "teacher_rshift" / "distill_meta.csv")
        regs_b = regressors(main_ / "teacher" / "distill_meta.csv")
        out["rshift_seeds"]["regressors_rmse_vs_teacher2"] = {k: rmse(v, p_r42["B_shift"]) for k, v in regs_b2.items()}

    # 3. other memories
    mems = {}
    for sub in ("r0_n256", "r1_n256", "r2_n256"):
        mem_rows = BY_ID.loc[SUBSETS[sub]["ids"]].reset_index()
        mean_pe = float(mem_rows[TARGET].mean())
        if sub == "r0_n256":
            p = {**p_main, **{k: v for k, v in p_r42.items() if k in ("B_shift",)}}
            cs = [p_main.get(f"C_s{s}") for s in (42, 43, 44)]
            td, ts, c_r, c_base = main_ / "teacher", ctrl / "teacher_rshift", p_r42.get("C_rshift"), p_main.get("C_s42")
        else:
            d = OUT / "robust" / sub
            p = preds(d / "final")
            if not p:
                continue
            cs = [p.get(f"C_s{s}") for s in (42, 43, 44)]
            td, ts, c_r, c_base = d / "teacher", d / "teacher_rshift", p.get("C_rshift_s42"), p.get("C_s42")
        cs = [c for c in cs if c is not None]
        rm_mean = rmse(np.full_like(YT, mean_pe))
        c_rmses = [rmse(c) for c in cs]
        ols_m = LinearRegression().fit(mem_rows[FEATURES].to_numpy(), mem_rows[TARGET].to_numpy()).predict(XT)
        rec = {"B": rmse(p["B"]), "B_shuffle": rmse(p["B_shuffle"]), "C": msd(c_rmses),
               "recovery_R": msd([(rm_mean - r) / (rm_mean - rmse(p["B"])) for r in c_rmses]),
               "C_corr_with_B": msd([agree(c, p["B"])["corr"] for c in cs]),
               "mean_baseline": rm_mean, "OLS_on_M_true": rmse(ols_m),
               "OLS_on_teacher_answers": rmse(regressors(td / "distill_meta.csv")["OLS"])}
        if c_r is not None and "B_shift" in p:
            thr = shifted_pe(mem_rows, 10, True)[1]
            rec["rshift_s42"] = region(p["B_shift"], p["B"], c_r, c_base, thr,
                                       regressors(ts / "distill_meta.csv"), regressors(td / "distill_meta.csv"))
            rec["rshift_s42"]["AT_threshold"] = thr
        mems[sub] = rec
    out["memories"] = mems

    # 4. k sweep
    pk = preds(OUT / "robust" / "k_sweep" / "final")
    ks = {}
    for k in (16, 64, 256):
        runs = [p_main.get(f"C_s{s}") for s in (42, 43, 44)] if k == 64 else [pk.get(f"C_k{k}_s{s}") for s in (42, 43, 44)]
        runs = [r for r in runs if r is not None]
        if runs:
            ks[k] = {"compression": 10979 / k, "test_rmse": msd([rmse(r) for r in runs]),
                     "abs_diff_with_B": msd([agree(r, p_main["B"])["abs_diff"] for r in runs])}
    out["k_sweep"] = ks

    (ROOT / "results" / "robust_summary.json").write_text(json.dumps(out, indent=2))
    f = lambda d: f"{d['mean']:.2f} ± {d['sd']:.2f} (n={d['n']})" if d["sd"] is not None else f"{d['mean']:.2f} (n=1)"  # noqa: E731
    L = ["## 1. 关键对照，3 个种子", "| 对照 | 老师 | 测试 RMSE | 与老师相关 | 与老师数字差 |", "|---|---|---|---|---|"]
    for n, r in out["control_seeds"].items():
        L.append(f"| {n} | {r['teacher']} | {f(r['test_rmse'])} | {f(r['corr_with_teacher'])} | {f(r['abs_diff_with_teacher'])} |")
    if "rshift_seeds" in out:
        r = out["rshift_seeds"]
        L += ["", "## 2. 区域平移（低温区 +10），3 个种子",
              f"- 老师平移：低温 {r['teacher_low']:+.2f}，高温 {r['teacher_high']:+.2f}",
              f"- C 平移：低温 {f(r['C_low'])}，高温 {f(r['C_high'])}",
              f"- 与 B'' 的 RMSE：C {f(r['C_rmse_vs_teacher2'])}；" +
              "，".join(f"{k} {v:.2f}" for k, v in r["regressors_rmse_vs_teacher2"].items())]
    L += ["", "## 3. 换记忆", "| 记忆 | B | B-打乱 | C（3 种子） | 恢复率 R | C 与 B 相关 | OLS（M 真值） | OLS（老师答案） | 区域平移 老师 低/高 | C 低/高 | 与 B'' RMSE：C / OLS / 5-NN |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sub, r in out["memories"].items():
        rs = r.get("rshift_s42")
        rs_txt = (f"{rs['teacher']['low']:+.2f} / {rs['teacher']['high']:+.2f} | {rs['C']['low']:+.2f} / {rs['C']['high']:+.2f} | "
                  f"{rs['C_rmse_vs_teacher2']:.2f} / {rs['OLS_rmse_vs_teacher2']:.2f} / {rs['5-NN_rmse_vs_teacher2']:.2f}") if rs else "— | — | —"
        L.append(f"| {sub} | {r['B']:.2f} | {r['B_shuffle']:.2f} | {f(r['C'])} | {f(r['recovery_R'])} | {f(r['C_corr_with_B'])} | "
                 f"{r['OLS_on_M_true']:.2f} | {r['OLS_on_teacher_answers']:.2f} | {rs_txt} |")
    L += ["", "## 4. 笔记大小", "| k | 压缩倍数 | 测试 RMSE | 与 B 数字差 |", "|---|---|---|---|"]
    for k, r in out["k_sweep"].items():
        L.append(f"| {k} | {r['compression']:.1f}× | {f(r['test_rmse'])} | {f(r['abs_diff_with_B'])} |")
    (ROOT / "results" / "robust_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
