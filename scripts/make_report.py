"""Collect every result file into results/report.md (tables only; no new model computation)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import FEATURES, ROOT, TARGET, load_splits  # noqa: E402
from fill_record import refusal_rate  # noqa: E402

R = ROOT / "results"
M = ROOT / "outputs" / "main"
SUBSET = "r0_n256"


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def table(df, cols, nd=3):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |\n" for _, r in df.iterrows())


def classic_on(split, rows, subsets):
    by_id = rows.set_index("id")
    mem = by_id.loc[subsets[SUBSET]["ids"]]
    q = rows[rows["split"] == split]
    X, y, Xq, yq = mem[FEATURES].to_numpy(), mem[TARGET].to_numpy(), q[FEATURES].to_numpy(), q[TARGET].to_numpy()
    sc = StandardScaler().fit(X)
    out = {}
    for name, p in {"均值底线": np.full(len(yq), y.mean()),
                    "线性回归 (OLS)": LinearRegression().fit(X, y).predict(Xq),
                    "5-NN": KNeighborsRegressor(5).fit(sc.transform(X), y).predict(sc.transform(Xq))}.items():
        out[name] = (float(np.sqrt(np.mean((p - yq) ** 2))), float(np.mean(np.abs(p - yq))))
    return out


def main():
    rows, subsets = load_splits()
    lines = ["# CCPP 记忆压缩实验 · 第一轮结果（Qwen3-4B，自动生成）\n"]

    # gate
    g = R / "gate" / "gate_by_length.csv"
    if g.exists():
        df = pd.read_csv(g)
        lines.append("## 1. 教师检查（tune 集 256 条；每个记忆长度 3 份随机抽样，均值）\n")
        lines.append(table(df, ["cond", "n_rows", "mem_tokens", "parse_rate", "rmse_mean", "rmse_sd", "mae_mean",
                                "mean_rmse", "ols_rmse", "knn5_rmse"], 2))
        a = R / "gate" / "gate_summary.csv"
        s = pd.read_csv(a)
        main_b = s[(s["cond"] == "B") & (s["subset"] == SUBSET)].iloc[0]
        lines.append(f"\n预注册主记忆 {SUBSET}：B RMSE {main_b['rmse_filled']:.2f}；相对 B-打乱 ΔRMSE "
                     f"{main_b['dRMSE_vs_shuf']:.2f} [95% CI {main_b['dRMSE_vs_shuf_lo']:.2f}, {main_b['dRMSE_vs_shuf_hi']:.2f}]。\n")
    al = M / "A_long_tune.csv"
    if al.exists():
        d = pd.read_csv(al)
        lines.append(f"\nA（无记忆）：16 词元内不输出数字；放宽到 384 词元后 {refusal_rate(d):.0%} 明确拒答，文中出现的数字是复述的"
                     "输入特征（如 RH），不是预测。A 按无预测处理，RMSE 为记忆池均值回退（≈ 均值底线）。\n")

    # checks
    eq = load_json(R / "gate" / "equivalence.json")
    cc = load_json(R / "cartridges_checks.json")
    p4 = load_json(R / "p4_check.json")
    lines.append("\n## 2. 实现检查\n")
    if eq:
        f32 = eq.get("fp32_vs_bf16_full", {})
        lines.append(f"- HF 前缀 KV 与整串前向：KL 均值 {eq['cached_vs_full_bf16']['kl_mean']['mean']:.4f}，argmax 一致 "
                     f"{eq['cached_vs_full_bf16']['argmax_agree']['mean']:.3f}；bf16 噪声底（fp32 vs bf16）KL "
                     f"{f32.get('kl_mean', {}).get('mean', float('nan')):.4f}；crop 后重跑完全一致：{eq['repeat_after_crop']['same_tokens']}\n")
    if cc:
        lines.append(f"- Cartridges(Flex) vs HF：k=64 初始化 KL {cc['flex_vs_hf_init64']['kl_mean']:.4f}；完整 M KL "
                     f"{cc['flex_vs_hf_fullM']['kl_mean']:.4f}\n")
        lines.append(f"- 存/读：冻结 {cc['cache_k64']['loaded_num_frozen']}、可训练 {cc['cache_k64']['loaded_num_trainable']}，"
                     f"数值一致 {cc['cache_k64']['roundtrip_identical']}（未打补丁会读成冻结 {cc['cache_k64']['unpatched_would_read_num_frozen']}）\n")
        for r in cc["adam_update"]:
            z = [s["frac_intended_nonzero_but_rounded_to_zero"] for s in r["steps"]]
            lines.append(f"- 原版 bf16 参数，lr={r['lr']}：3 步中被舍入为 0 的更新比例 {min(z):.3f}–{max(z):.3f}\n")
    if p4:
        for r in p4["adam_update_fp32"]:
            z = [s["frac_intended_nonzero_but_rounded_to_zero"] for s in r["steps"]]
            lines.append(f"- P4 fp32 主参数，lr={r['lr']}：被舍入比例 ≤ {max(z):.2e}，前向 logits 最大差 {p4['forward']['max_abs_logit_diff']}\n")

    td = load_json(M / "teacher" / "summary.json")
    if td:
        lines.append("\n## 3. 教师数据\n")
        for k in ["distill", "dev"]:
            d = td[k]
            lines.append(f"- {k}: n={d['n']}，解析率 {d['parse_rate']:.3f}，教师 RMSE {d['teacher_rmse']:.2f}，"
                         f"top-20 覆盖最小值 {d['topk_mass_min']:.6f}\n")

    # training / selection
    runs = sorted((M / "runs").glob("lr*_s*")) if (M / "runs").exists() else []
    if runs:
        lines.append("\n## 4. 训练与开发集选择（全词表 KL(B‖C)，沿教师答案路径）\n")
        recs = []
        for r in runs:
            s = load_json(r / "dev_selection.json")
            if s:
                dg = s.get("best_dev_generation", {})
                recs.append({"run": r.name, "init_dev_kl": s["init_dev_kl"], "best_dev_kl": s["best_dev_kl"],
                             "best_updates": s["best_updates"], "best_dev_rmse": dg.get("rmse_vs_truth_parsed"),
                             "best_dev_absdiff_B": dg.get("abs_diff_vs_teacher"), "best_dev_parse": dg.get("parse_rate")})
        lines.append(table(pd.DataFrame(recs), list(recs[0].keys()), 4))
        lr = (M / "chosen_lr.txt").read_text().strip() if (M / "chosen_lr.txt").exists() else "?"
        lines.append(f"\n选中学习率：{lr}\n")

    # final test table
    fs = M / "final" / "final_test_summary.csv"
    if fs.exists():
        s = pd.read_csv(fs)
        lines.append("\n## 5. 测试集结果（2000 条；B 完整记忆 10,979 词元，C 为 k=64）\n")
        lines.append(table(s, ["cond", "cache_positions", "cache_MB", "parse_rate", "rmse_filled", "mae_filled",
                               "abs_diff_vs_B", "n_both_parsed", "kl_B_X", "top1_agree_B_X"], 3))
        c = s[s["cond"].str.match(r"C_s\d+")]
        if len(c):
            lines.append(f"\nC-训练后（{len(c)} 个种子）：RMSE {c['rmse_filled'].mean():.3f} ± {c['rmse_filled'].std(ddof=1):.3f}；"
                         f"MAE {c['mae_filled'].mean():.3f} ± {c['mae_filled'].std(ddof=1):.3f}；"
                         f"KL(B‖C) {c['kl_B_X'].mean():.4f} ± {c['kl_B_X'].std(ddof=1):.4f}；"
                         f"与 B 数字差 {c['abs_diff_vs_B'].mean():.3f} ± {c['abs_diff_vs_B'].std(ddof=1):.3f}\n")
        lines.append(f"\n压缩率（完整 M 词元数 ÷ k）：10979 / 64 = {10979 / 64:.1f}×\n")
        pq = M / "final" / "final_test_per_query.csv"
        if pq.exists():
            d = pd.read_csv(pq)
            b = d[d["cond"] == "B"].set_index("qid")["text"]
            same = {c: float((g.set_index("qid")["text"] == b.reindex(g["qid"]).values).mean())
                    for c, g in d[d["cond"].str.match(r"C_s\d+")].groupby("cond")}
            lines.append("\n注：top1_agree_B_X 是沿 B 答案路径（teacher forcing）逐词元 argmax 一致率，包含几乎必然一致的“.”与结束符；"
                         "整条答案与 B 完全相同的比例：" + "，".join(f"{c} {v:.1%}" for c, v in same.items()) + "。\n")
        at = M / "A_long_test.csv"
        if at.exists():
            d = pd.read_csv(at)
            pool = float(rows[rows["split"] == "memory_pool"][TARGET].mean())
            yt = rows[rows["split"] == "test"][TARGET].to_numpy()
            lines.append(f"\nA（无记忆）test：不给出数值预测（长回答中 {refusal_rate(d):.0%} 明确拒答）；按均值回退 RMSE "
                         f"{np.sqrt(np.mean((pool - yt) ** 2)):.3f}，MAE {np.mean(np.abs(pool - yt)):.3f}\n")
        lines.append("\n传统方法（拟合同一份 256 行记忆）test：\n")
        for k, (rm, ma) in classic_on("test", rows, subsets).items():
            lines.append(f"- {k}：RMSE {rm:.3f}，MAE {ma:.3f}\n")

    st = load_json(R / "test_stats.json")
    if st:
        p, rr = st["pairs"], st["recovery_R"]
        lines.append("\n## 6. 配对 bootstrap（测试集，按 qid 配对，5000 次，95% CI）\n")
        for k in ["C_s42 - B", "C_s43 - B", "C_s44 - B", "C_avg3 - B", "C_avg3 - B_shuffle", "C_avg3 - mean", "B - mean", "B - B_shuffle"]:
            lines.append(f"- RMSE({k.split(' - ')[0]}) − RMSE({k.split(' - ')[1]}) = {p[k]['diff']:+.3f} [{p[k]['ci95'][0]:+.3f}, {p[k]['ci95'][1]:+.3f}]\n")
        lines.append(f"- 恢复率 R = (RMSE_mean − RMSE_C) / (RMSE_mean − RMSE_B) = {rr['point']:.1%} [{rr['ci95'][0]:.1%}, {rr['ci95'][1]:.1%}]"
                     "（A 不给数字，以均值底线为无记忆参照）\n")
    from fill_record import gather, instability  # noqa: E402
    notes = [t for t in (instability(r) for r in gather()["runs"].values()) if t]
    if notes:
        lines.append("\n## 7. 训练稳定性\n")
        lines += [f"- {t}\n" for t in notes]
    ps = load_json(R / "pseudo_label_control.json")
    if ps:
        lines.append("\n## 9. 信息来源对照：C 是在压缩 M，还是在学 1024 个教师伪标签？\n")
        lines.append("不用任何记忆，只用 C 训练时同样的 1024 个 (X, 教师答案) 拟合传统回归器，在 test 2000 条上评估：\n\n")
        rows_ = [{"方法": f"{k}（教师伪标签）", **v} for k, v in ps["teacher_pseudo_labels"].items()]
        rows_ += [{"方法": f"{k}（真值标签，参照）", **v} for k, v in ps["true_labels_reference"].items()]
        cr = ps["C_reference"]
        rows_.append({"方法": "C（64 位置 Z，3 种子平均）", "test_rmse_vs_truth": cr["test_rmse_mean_of_seed_rmses"],
                      "corr_with_B": cr["corr_with_B_mean_of_seeds"], "mean_abs_diff_vs_B": cr["mean_abs_diff_vs_B_mean_of_seeds"]})
        lines.append(table(pd.DataFrame(rows_), ["方法", "test_rmse_vs_truth", "corr_with_B", "mean_abs_diff_vs_B"], 3))
        lines.append("\n结论：RMSE 层面，用同样伪标签的 OLS（8.80）已不差于 C 和 B，因此“保留 97% 收益”不能单独证明 C 压缩了 M；"
                     "但 C 比这些回归器更贴近 B（相关 0.953 vs ≤0.940，数字差 1.66 vs ≥2.61 MW），说明它复现的是 B 的具体行为。\n")

    lines.append("""
## 10. 主结论（建议写法）与解读限定

**主结论**：在 CCPP / Qwen3-4B / 单份 256 行记忆上，按开发集 KL 早停蒸馏得到的 64 位置前缀（KV 位置为完整记忆的 1/171.5），
在同分布测试查询上复现了 B 的大部分行为：test RMSE 10.01 ± 0.19（3 种子）对 B 9.79，略差且显著（+0.22 [0.05, 0.40]）；
相对均值底线的改进保留 97% [94.6%, 99.3%]（只含查询抽样的不确定性）。B 本身近似 1-NN 抄表（OLS 4.59），
且蒸馏查询与测试同分布，尚不能排除 C 只是从 1024 个伪标签学到了回归（见第 9 节）。

**限定语**（汇报时请保留）：
1. 单一设定：CCPP、Qwen3-4B、一份 256 行记忆（r0_n256，三份抽样中教师最弱的一份，未挑选）、k=64、1024 条蒸馏、贪心解码。
2. B 不是上限：B 与 1-NN 相关 0.85，约 57–61% 的预测恰为 M 中某个 PE；R=97% 衡量的是对一个弱教师的保真度。
3. 不参与选学习率的两个种子都显著差于 B（s43 +0.29，s44 +0.36）；s42 与 B 无差异（+0.01），但 s42 参与了学习率选择。
4. C-初始与 A 都不输出数字，两行实际是均值回退；KL(B‖C) 与 top-1 一致率沿 B 的贪心答案逐词元计算，整条答案与 B 完全相同只有 39–44%。
5. 学习率二选一的开发 KL 差（0.0085）小于种子间差（0.016）；lr=2e-2 下 s44 中途发散，靠早停选到发散前的 checkpoint。
6. 171.5× 是 KV 位置数之比（MiB 为理论值，未实测显存与延迟）；训练成本未计入（教师数据约 4 分钟，每个 run 约 15 分钟 H100）。
7. 实现为 Cartridges + 补丁 P1/P2/P4（fp32 主参数），偏离原版 bf16 参数；P4 只做了 3 步数值检查，未在完整训练上消融。
8. 16 行记忆不通过是输出格式问题（16 词元内解析率 1.6–6.3%），不是记忆无效。

## 11. 下一步（按优先级）
1. **信息来源对照**：用“打乱教师”（B-打乱生成的伪标签）与“无关文本初始化”各训一个 C；蒸馏量扫描 {64, 256, 1024}；再加一组只由 M 生成、与 test 查询无关的问题（接近原版 self-study），看 test 上还剩多少。
2. **稳健性**：r0/r1/r2_n256 与 r0_n512（B 在 tune 上 6.12，接近 5-NN）× k∈{16, 64, 256} × 3 种子；学习率按 3 种子平均开发 KL 选，2e-2 加 warmup/梯度裁剪；看 B 变强时 C 是否同步变强。
3. **口径与成本**：去掉 P4 的消融；沿 C 自己答案的 KL 与整条答案一致率；B（前缀缓存）与 C 的实测显存、延迟、吞吐，以及训练成本回本所需查询数。
""")
    figs = [f for f in ["fig_kl_curves.png", "fig_test_rmse.png"] if (R / f).exists()]
    if figs:
        lines.append("\n## 8. 图\n")
        lines += [f"![{f}]({f})\n" for f in figs]

    out = R / "report.md"
    out.write_text("".join(lines))
    print(out.read_text())


if __name__ == "__main__":
    main()
