"""Control for "is C compressing M, or regressing on the 1024 teacher pseudo-labels?" (CPU only).

Fits classic regressors on the same 1024 distillation inputs X with the teacher's answers B(X) as targets
(exactly the supervision C was trained on, minus the memory), and scores them on the test split against
the truth and against B's / C's test predictions. Also reports the same regressors fit on the true PE of
those 1024 rows (labels C never saw) as an upper reference. Writes results/pseudo_label_control.json.
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
from ccpp_common import FEATURES, ROOT, TARGET, load_splits  # noqa: E402


def main():
    rows, _ = load_splits()
    by_id = rows.set_index("id")
    meta = pd.read_csv(ROOT / "outputs" / "main" / "teacher" / "distill_meta.csv").dropna(subset=["teacher_pred"])
    X = by_id.loc[meta["qid"], FEATURES].to_numpy()
    y_teacher = meta["teacher_pred"].to_numpy()
    y_true = by_id.loc[meta["qid"], TARGET].to_numpy()
    test = rows[rows["split"] == "test"].set_index("id")
    pq = pd.read_csv(ROOT / "outputs" / "main" / "final" / "final_test_per_query.csv")
    preds = {c: g.set_index("qid").loc[test.index, "pred_filled"].to_numpy() for c, g in pq.groupby("cond") if c in ("B", "C_s42", "C_s43", "C_s44")}
    Xt, yt = test[FEATURES].to_numpy(), test[TARGET].to_numpy()
    sc = StandardScaler().fit(X)

    def models():
        return {"OLS": LinearRegression(), "1-NN": KNeighborsRegressor(1), "5-NN": KNeighborsRegressor(5)}

    out = {"n_train": len(meta), "n_test": len(test)}
    for label, target in [("teacher_pseudo_labels", y_teacher), ("true_labels_reference", y_true)]:
        res = {}
        for name, m in models().items():
            Xf, Xq = (sc.transform(X), sc.transform(Xt)) if "NN" in name else (X, Xt)
            p = m.fit(Xf, target).predict(Xq)
            res[name] = {
                "test_rmse_vs_truth": float(np.sqrt(np.mean((p - yt) ** 2))),
                "corr_with_B": float(np.corrcoef(p, preds["B"])[0, 1]),
                "mean_abs_diff_vs_B": float(np.mean(np.abs(p - preds["B"]))),
            }
        out[label] = res
    c_avg = np.mean([preds[c] for c in ("C_s42", "C_s43", "C_s44")], axis=0)
    out["C_reference"] = {
        "test_rmse_mean_of_seed_rmses": float(np.mean([np.sqrt(np.mean((preds[c] - yt) ** 2)) for c in ("C_s42", "C_s43", "C_s44")])),
        "corr_with_B_mean_of_seeds": float(np.mean([np.corrcoef(preds[c], preds["B"])[0, 1] for c in ("C_s42", "C_s43", "C_s44")])),
        "mean_abs_diff_vs_B_mean_of_seeds": float(np.mean([np.mean(np.abs(preds[c] - preds["B"])) for c in ("C_s42", "C_s43", "C_s44")])),
        "seed_average_prediction_rmse": float(np.sqrt(np.mean((c_avg - yt) ** 2))),
    }
    out["B_reference"] = {"test_rmse": float(np.sqrt(np.mean((preds["B"] - yt) ** 2)))}
    (ROOT / "results" / "pseudo_label_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
