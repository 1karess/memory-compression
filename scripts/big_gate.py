"""Scale-up gate: does the teacher still use a 10x memory (~2.5k rows, ~110k tokens)? GPU.

Bigger memories need more rows than the 1024-row memory pool, so the pool is extended with the unused reserve split
(test / dev / tune / distill are untouched, so there is no leakage). Memory r{r}_n{n}: the existing r{r}_n512 rows
first, then the rest of memory_pool + reserve in a fixed random order (nested, like the existing subsets).
Beyond the native 40,960 positions the model needs YaRN (factor 4, as recommended for Qwen3); n512 is also run without
YaRN to measure what YaRN itself costs. Conditions on the 256 tune queries: B and B-shuffle per memory size.
Outputs results/big_gate/{big_subsets.json, gate_per_query.csv, gate_summary.csv}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import FEATURES, MODEL_DIR, ROOT, TARGET, build_ids, load_splits, memory_text, parse_number, user_text  # noqa: E402
from teacher_gate import greedy  # noqa: E402

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def big_subsets(rows, subsets, sizes, resamples):
    pool = rows.loc[rows["split"].isin(["memory_pool", "reserve"]), "id"].to_numpy()
    out = {}
    for r in range(resamples):
        head = subsets[f"r{r}_n512"]["ids"]
        rest = np.setdiff1d(pool, head)
        order = head + np.random.default_rng(200 + r).permutation(rest).tolist()
        for n in sizes:
            ids = order[:n]
            pe = rows.set_index("id").loc[ids, TARGET].to_numpy()
            out[f"r{r}_n{n}"] = {"ids": [int(i) for i in ids],
                                 "shuffled_pe": np.random.default_rng(3000 + 10 * r + n).permutation(pe).tolist()}
    return out


def load_model(yarn):
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    if yarn:
        cfg.rope_scaling, cfg.max_position_embeddings = dict(YARN), 131072
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, config=cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.cuda().eval()


@torch.inference_mode()
def prefill_chunked(model, mem_ids, chunk=8192):
    cache = DynamicCache()
    for i in range(0, len(mem_ids), chunk):
        model(input_ids=torch.tensor([mem_ids[i:i + chunk]], device="cuda"), past_key_values=cache, use_cache=True,
              logits_to_keep=1)
    assert cache.get_seq_length() == len(mem_ids)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "big_gate"))
    ap.add_argument("--sizes", default="512,1024,2048,2560")
    ap.add_argument("--resamples", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=16)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    big = big_subsets(rows, subsets, sizes, args.resamples)
    (out_dir / "big_subsets.json").write_text(json.dumps(big))
    by_id = rows.set_index("id")
    tune = rows[rows["split"] == "tune"].reset_index(drop=True)

    runs = [(False, f"r{r}_n512") for r in range(args.resamples)] + \
           [(True, f"r{r}_n{n}") for n in sizes for r in range(args.resamples)]
    per_query, model, model_yarn = [], None, None
    for yarn, key in runs:
        if model is None or model_yarn != yarn:
            del model
            torch.cuda.empty_cache()
            model, model_yarn = load_model(yarn), yarn
        mem_rows = by_id.loc[big[key]["ids"]].reset_index()
        for cond in ("B", "B_shuffle"):
            t0 = time.time()
            ys = big[key]["shuffled_pe"] if cond == "B_shuffle" else mem_rows[TARGET].tolist()
            mem = memory_text(mem_rows, ys if cond == "B_shuffle" else None)
            cache, mem_len = None, None
            for _, q in tune.iterrows():
                mem_ids, tail = build_ids(tok, mem, user_text(q))
                if cache is None:
                    cache, mem_len = prefill_chunked(model, mem_ids), len(mem_ids)
                gen, _ = greedy(model, cache, mem_len, tail, args.max_new)
                pred = parse_number(tok.decode(gen, skip_special_tokens=True))
                per_query.append({"yarn": yarn, "subset": key, "n_rows": len(mem_rows), "cond": cond, "mem_tokens": mem_len,
                                  "qid": int(q["id"]), "y": float(q[TARGET]), "pred": pred, "parse_ok": pred is not None,
                                  "pred_filled": pred if pred is not None else float(np.mean(ys))})
            del cache
            torch.cuda.empty_cache()
            print(f"[big_gate] yarn={yarn} {key} {cond} mem_tokens={mem_len} {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(per_query).to_csv(out_dir / "gate_per_query.csv", index=False)

    df = pd.DataFrame(per_query)
    summ = []
    for (yarn, key, cond), g in df.groupby(["yarn", "subset", "cond"], sort=False):
        rec = {"yarn": yarn, "subset": key, "cond": cond, "n_rows": int(g["n_rows"].iloc[0]),
               "mem_tokens": int(g["mem_tokens"].iloc[0]), "parse_rate": float(g["parse_ok"].mean()),
               "rmse": float(np.sqrt(((g["pred_filled"] - g["y"]) ** 2).mean()))}
        if cond == "B":  # classic regressors fitted on the same memory rows, for scale
            m = by_id.loc[big[key]["ids"]]
            X, y, Xq = m[FEATURES].to_numpy(), m[TARGET].to_numpy(), tune[FEATURES].to_numpy()
            sc = StandardScaler().fit(X)
            rec["ols"] = float(np.sqrt(((LinearRegression().fit(X, y).predict(Xq) - tune[TARGET]) ** 2).mean()))
            rec["knn5"] = float(np.sqrt(((KNeighborsRegressor(5).fit(sc.transform(X), y).predict(sc.transform(Xq))
                                          - tune[TARGET]) ** 2).mean()))
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "gate_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
