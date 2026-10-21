"""Step 1: load CCPP (Sheet1 only), split by formatted input x, build memory subsets, run classic baselines.

Outputs (data/processed/):
  rows.csv              id, AT, V, AP, RH, PE, x_str, split
  memory_subsets.json   {"r{r}_n{n}": {"ids": [...], "shuffled_pe": [...]}} (nested prefixes per resample)
  split_report.json     sizes, duplicate stats
results/baselines_tune.csv  mean / OLS / 5-NN fitted on each memory subset, scored on the tune split
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
from ccpp_common import FEATURES, PROCESSED, ROOT, TARGET, fmt_x  # noqa: E402

SEED = 0
SPLIT_SIZES = [("memory_pool", 1024), ("distill", 1024), ("tune", 256), ("dev", 256), ("test", 2000)]
MEM_LENGTHS = [16, 64, 128, 256, 512]
N_RESAMPLES = 3


def load_ccpp() -> pd.DataFrame:
    xlsx = next((ROOT / "data" / "raw").rglob("Folds5x2_pp.xlsx"))
    # The workbook holds 5 shuffled copies of the same data; use only the first sheet.
    df = pd.read_excel(xlsx, sheet_name=0)
    df = df[FEATURES + [TARGET]].astype(float).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    return df


def split_by_x(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["x_str"] = [fmt_x(r) for _, r in df.iterrows()]
    groups = df.groupby("x_str")["id"].apply(list).tolist()
    order = rng.permutation(len(groups))
    split = {}
    targets = iter(SPLIT_SIZES)
    name, size = next(targets)
    count = 0
    for gi in order:
        if name != "reserve" and count >= size:
            try:
                name, size = next(targets)
            except StopIteration:
                name, size = "reserve", None
            count = 0
        for i in groups[gi]:
            split[i] = name
        count += len(groups[gi])
    df["split"] = df["id"].map(split)
    return df


def baselines(mem: pd.DataFrame, query: pd.DataFrame) -> dict:
    X, y = mem[FEATURES].to_numpy(), mem[TARGET].to_numpy()
    Xq, yq = query[FEATURES].to_numpy(), query[TARGET].to_numpy()
    scaler = StandardScaler().fit(X)
    preds = {
        "mean": np.full(len(yq), y.mean()),
        "ols": LinearRegression().fit(X, y).predict(Xq),
        "knn5": KNeighborsRegressor(n_neighbors=min(5, len(y))).fit(scaler.transform(X), y).predict(scaler.transform(Xq)),
        "knn1": KNeighborsRegressor(n_neighbors=1).fit(scaler.transform(X), y).predict(scaler.transform(Xq)),
    }
    out = {}
    for k, p in preds.items():
        err = p - yq
        out[f"{k}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{k}_mae"] = float(np.mean(np.abs(err)))
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_ccpp()
    df = split_by_x(df, rng)

    dup_groups = df.groupby("x_str").filter(lambda g: len(g) > 1)
    conflicting = dup_groups.groupby("x_str")[TARGET].nunique()
    report = {
        "n_rows": int(len(df)),
        "n_unique_x": int(df["x_str"].nunique()),
        "n_rows_in_duplicate_x_groups": int(len(dup_groups)),
        "n_duplicate_x_groups_with_different_y": int((conflicting > 1).sum()),
        "split_sizes": df["split"].value_counts().to_dict(),
        "x_leak_across_splits": int((df.groupby("x_str")["split"].nunique() > 1).sum()),
        "seed": SEED,
    }
    assert report["x_leak_across_splits"] == 0
    df.to_csv(PROCESSED / "rows.csv", index=False)

    pool = df[df["split"] == "memory_pool"]["id"].to_numpy()
    subsets = {}
    for r in range(N_RESAMPLES):
        perm = np.random.default_rng(100 + r).permutation(pool)
        for n in MEM_LENGTHS:
            ids = perm[:n].tolist()
            pe = df.set_index("id").loc[ids, TARGET].to_numpy()
            shuffled = np.random.default_rng(1000 + 10 * r + n).permutation(pe)
            subsets[f"r{r}_n{n}"] = {
                "ids": ids,
                "shuffled_pe": shuffled.tolist(),
                "frac_pe_unchanged_by_shuffle": float(np.mean(shuffled == pe)),
            }
    (PROCESSED / "memory_subsets.json").write_text(json.dumps(subsets))
    (PROCESSED / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    tune = df[df["split"] == "tune"]
    by_id = df.set_index("id")
    rows = []
    for key, s in subsets.items():
        r, n = key.split("_")
        mem = by_id.loc[s["ids"]].reset_index()
        rows.append({"subset": key, "resample": int(r[1:]), "n_rows": int(n[1:]), **baselines(mem, tune)})
    full = df[df["split"].isin(["memory_pool", "distill"])]
    rows.append({"subset": "memory_pool+distill", "resample": -1, "n_rows": len(full), **baselines(full, tune)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "baselines_tune.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
