"""Large memories (n = 1024, 2048, 2560 rows) for the scale-up, CPU only. rows.csv and the splits are unchanged.

The rows come from memory_pool + reserve (6032 rows, never used as queries). Resample r keeps its existing order for
the first 512 rows (so r{r}_n256 and r{r}_n512 are prefixes of the large memories) and continues with a fresh
permutation of the remaining memory_pool + reserve rows. B-shuffle labels as in prepare_data.py.

Outputs:
  data/processed/memory_subsets_large.json  {"r{r}_n{n}": {"ids", "shuffled_pe", "frac_pe_unchanged_by_shuffle"}}
  results/baselines_tune_large.csv          mean / OLS / 5-NN / 1-NN fitted on each large memory, scored on tune
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import PROCESSED, ROOT, TARGET  # noqa: E402
from prepare_data import N_RESAMPLES, baselines  # noqa: E402

LENGTHS = [1024, 2048, 2560]
PREFIX = 512


def main():
    rows = pd.read_csv(PROCESSED / "rows.csv")
    small = json.loads((PROCESSED / "memory_subsets.json").read_text())
    by_id = rows.set_index("id")
    source = rows.loc[rows["split"].isin(["memory_pool", "reserve"]), "id"].to_numpy()
    query_ids = set(rows.loc[~rows["split"].isin(["memory_pool", "reserve"]), "id"])

    subsets = {}
    for r in range(N_RESAMPLES):
        head = small[f"r{r}_n{PREFIX}"]["ids"]
        rest = np.random.default_rng(200 + r).permutation(np.setdiff1d(source, head))
        order = list(head) + rest.tolist()
        assert len(set(order)) == len(order) and not set(order) & query_ids
        for n in LENGTHS:
            ids = [int(i) for i in order[:n]]
            pe = by_id.loc[ids, TARGET].to_numpy()
            shuffled = np.random.default_rng(1000 + 10 * r + n).permutation(pe)
            subsets[f"r{r}_n{n}"] = {"ids": ids, "shuffled_pe": shuffled.tolist(),
                                     "frac_pe_unchanged_by_shuffle": float(np.mean(shuffled == pe))}
        assert subsets[f"r{r}_n{LENGTHS[0]}"]["ids"][:256] == small[f"r{r}_n256"]["ids"]
    (PROCESSED / "memory_subsets_large.json").write_text(json.dumps(subsets))

    tune = rows[rows["split"] == "tune"]
    out = []
    for key, s in subsets.items():
        r, n = key.split("_")
        out.append({"subset": key, "resample": int(r[1:]), "n_rows": int(n[1:]), **baselines(by_id.loc[s["ids"]].reset_index(), tune)})
    for r in range(N_RESAMPLES):
        a, b = set(subsets[f"r{r}_n2560"]["ids"]), set(subsets[f"r{(r + 1) % N_RESAMPLES}_n2560"]["ids"])
        print(f"overlap r{r} / r{(r + 1) % N_RESAMPLES} at n2560: {len(a & b)} rows")
    out = pd.DataFrame(out)
    out.to_csv(ROOT / "results" / "baselines_tune_large.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
