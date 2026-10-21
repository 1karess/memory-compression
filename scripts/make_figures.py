"""Figures for the meeting: KL curves per seed and the test RMSE comparison. Writes results/fig_*.png.

Palette: reference categorical slots 1-3 (validated all-pairs, light mode) for the three seeds; reference
lines and bars for non-seed conditions stay in neutral ink so colour only ever means "which seed".
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import ROOT, load_splits  # noqa: E402
from fill_record import classic as classic_baselines, train_losses  # noqa: E402

MAIN = ROOT / "outputs" / "main"
OUT = ROOT / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3
SEEDS = [42, 43, 44]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def end_labels(ax, ends, min_gap=0.075):
    """Direct labels at the line ends, nudged apart in log10 space so they never overlap."""
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        ly = np.log10(y)
        if placed and ly - placed[-1] < min_gap:
            ly = placed[-1] + min_gap
        placed.append(ly)
        ax.annotate(text, (x, y), xytext=(x, 10 ** ly), textcoords="data", va="center", fontsize=9, color=INK2,
                    xycoords="data", annotation_clip=False)


def kl_curves(lr):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, which in zip(axes, ["dev", "train"]):
        ends = []
        for color, sd in zip(SERIES, SEEDS):
            run = MAIN / "runs" / f"lr{lr}_s{sd}"
            df = pd.read_csv(run / "dev_selection.csv")
            if which == "dev":
                x, y = df["updates"], df["dev_kl"]
                ax.plot(x, y, color=color, lw=2, label=f"seed {sd}")
                best = json.loads((run / "dev_selection.json").read_text())
                b = df[df["ckpt"] == best["best_ckpt"]].iloc[0]
                ax.plot(b["updates"], b["dev_kl"], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
                ends.append((x.iloc[-1] + 12, y.iloc[-1], f"seed {sd}"))
                if df["dev_kl"].iloc[1:].max() > 3 * best["best_dev_kl"] and df["dev_kl"].idxmax() > 1:
                    peak = df.loc[df["dev_kl"].iloc[1:].idxmax()]
                    ax.annotate(f"seed {sd} destabilised;\nearlier checkpoint kept",
                                (peak["updates"], peak["dev_kl"]), xytext=(0.17, 0.45), textcoords="axes fraction",
                                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=2, shrinkB=4))
            else:
                losses = train_losses(f"lr{lr}_s{sd}")
                steps = sorted(losses)
                ax.plot(steps, [losses[s] for s in steps], color=color, lw=2, label=f"seed {sd}")
        ax.set_yscale("log")
        if ends:
            end_labels(ax, ends)
            ax.set_xlim(right=ax.get_xlim()[1] + 60)
        ax.set_xlabel("optimizer updates")
        if which == "dev":
            ax.set_title("Dev: full-vocabulary KL(B‖C) along the teacher's answer\n(dot = checkpoint selected per seed)",
                         loc="left", fontsize=10.5, color=INK)
            ax.set_ylabel("KL(B‖C), log scale")
        else:
            ax.set_title("Train: truncated top-20 soft-label CE (training log)\nnot a KL; trend only",
                         loc="left", fontsize=10.5, color=INK)
            ax.set_ylabel("CE, log scale")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(f"Training a 64-position cartridge Z on CCPP (Qwen3-4B, lr={lr}, 1024 distillation queries)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kl_curves.png", dpi=180)


def test_rmse():
    s = pd.read_csv(MAIN / "final" / "final_test_summary.csv").set_index("cond")
    stats = json.loads((OUT / "test_stats.json").read_text())
    rows_df, subsets = load_splits()
    classic = {k: v[0] for k, v in classic_baselines("test", rows_df, subsets).items()}
    rows = [("B: full memory M (10,979 positions)", s.loc["B", "rmse_filled"], "ref"),
            *[(f"C: cartridge Z, seed {sd} (64 positions)", s.loc[f"C_s{sd}", "rmse_filled"], i) for i, sd in enumerate(SEEDS)],
            ("B-shuffle: M with permuted PE", s.loc["B_shuffle", "rmse_filled"], "ref"),
            ("Mean of M / A (no memory) / C-init", stats["rmse"]["mean"], "ref"),
            ("Reference: linear regression fit on M", classic["ols"], "classic"),
            ("Reference: 5-NN on M", classic["knn5"], "classic")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (name, v, kind) in enumerate(rows):
        color = SERIES[kind] if isinstance(kind, int) else ("#a3a29d" if kind == "ref" else "#d6d5d0")
        ax.barh(i, v, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test RMSE of predicted PE (MW), 2000 queries, lower is better")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.12)
    r = stats["recovery_R"]
    fig.suptitle(f"Z keeps {r['point']:.0%} of the full-memory gain over the mean baseline "
                 f"(95% CI {r['ci95'][0]:.0%}–{r['ci95'][1]:.0%})\nwith 171.5× fewer KV positions; "
                 "classic regressors on the same 256 rows remain better than the teacher",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_test_rmse.png", dpi=180)


def main():
    lr = (MAIN / "chosen_lr.txt").read_text().strip()
    kl_curves(lr)
    test_rmse()
    print("wrote", OUT / "fig_kl_curves.png", OUT / "fig_test_rmse.png")


if __name__ == "__main__":
    main()
