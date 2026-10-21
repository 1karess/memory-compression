"""Scale-up summary (jobs/19), CPU only: 256-row vs 512-row memory for draws r0 / r1 / r2, and the k sweep on r0_n512.
Missing pieces are skipped. Writes results/scale_summary.json and results/scale_summary.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import FEATURES, ROOT, TARGET  # noqa: E402
from robust_report import BY_ID, OUT, SUBSETS, XT, YT, agree, msd, preds, regressors, rmse  # noqa: E402


def memory_row(p, cs, sub, teacher_dir):
    mem_rows = BY_ID.loc[SUBSETS[sub]["ids"]].reset_index()
    rm_mean = rmse(np.full_like(YT, float(mem_rows[TARGET].mean())))
    c_rmses = [rmse(c) for c in cs]
    ols_m = LinearRegression().fit(mem_rows[FEATURES].to_numpy(), mem_rows[TARGET].to_numpy()).predict(XT)
    return {"B": rmse(p["B"]), "B_shuffle": rmse(p["B_shuffle"]), "C": msd(c_rmses),
            "recovery_R": msd([(rm_mean - r) / (rm_mean - rmse(p["B"])) for r in c_rmses]),
            "C_corr_with_B": msd([agree(c, p["B"])["corr"] for c in cs]),
            "C_abs_diff_with_B": msd([agree(c, p["B"])["abs_diff"] for c in cs]),
            "OLS_on_M_true": rmse(ols_m), "OLS_on_teacher_answers": rmse(regressors(teacher_dir / "distill_meta.csv")["OLS"])}


def main():
    out = {"memories": {}, "k_sweep_r0_n512": {}}
    for r in ("r0", "r1", "r2"):
        # 256 rows: main run (r0) or jobs/16 (r1, r2)
        if r == "r0":
            p = preds(OUT / "main" / "final")
            cs, td = [p.get(f"C_s{s}") for s in (42, 43, 44)], OUT / "main" / "teacher"
        else:
            d = OUT / "robust" / f"{r}_n256"
            p = preds(d / "final")
            cs, td = [p.get(f"C_s{s}") for s in (42, 43, 44)], d / "teacher"
        cs = [c for c in cs if c is not None]
        if p and cs:
            out["memories"][f"{r}_n256"] = memory_row(p, cs, f"{r}_n256", td)
        d = OUT / "scale" / f"{r}_n512"
        p = preds(d / "final")
        cs = [c for c in (p.get(f"C_k64_s{s}") for s in (42, 43, 44)) if c is not None]
        if p and cs:
            out["memories"][f"{r}_n512"] = memory_row(p, cs, f"{r}_n512", d / "teacher")
        if r == "r0" and p:
            for k in (16, 64, 256):
                runs = [c for c in (p.get(f"C_k{k}_s{s}") for s in (42, 43, 44)) if c is not None]
                if runs:
                    out["k_sweep_r0_n512"][k] = {"test_rmse": msd([rmse(c) for c in runs]),
                                                 "abs_diff_with_B": msd([agree(c, p["B"])["abs_diff"] for c in runs])}

    (ROOT / "results" / "scale_summary.json").write_text(json.dumps(out, indent=2))
    f = lambda d: f"{d['mean']:.2f} ± {d['sd']:.2f} (n={d['n']})" if d["sd"] is not None else f"{d['mean']:.2f} (n=1)"  # noqa: E731
    L = ["## 1. 记忆 256 行 vs 512 行（k = 64）",
         "| 记忆 | B | B-打乱 | C（3 种子） | 恢复率 R | C 与 B 相关 | C 与 B 数字差 | OLS（M 真值） | OLS（老师答案） |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sub, r in out["memories"].items():
        L.append(f"| {sub} | {r['B']:.2f} | {r['B_shuffle']:.2f} | {f(r['C'])} | {f(r['recovery_R'])} | {f(r['C_corr_with_B'])} | "
                 f"{f(r['C_abs_diff_with_B'])} | {r['OLS_on_M_true']:.2f} | {r['OLS_on_teacher_answers']:.2f} |")
    L += ["", "## 2. 笔记大小（r0_n512）", "| k | 压缩倍数 | 测试 RMSE | 与 B 数字差 |", "|---|---|---|---|"]
    n_tok = None
    s = OUT / "scale" / "r0_n512" / "teacher" / "summary.json"
    if s.exists():
        n_tok = json.loads(s.read_text())["distill"]["mem_tokens"]
    for k, r in out["k_sweep_r0_n512"].items():
        comp = f"{n_tok / k:.1f}×" if n_tok else "—"
        L.append(f"| {k} | {comp} | {f(r['test_rmse'])} | {f(r['abs_diff_with_B'])} |")
    (ROOT / "results" / "scale_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
