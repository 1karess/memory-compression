"""Step 2: teacher gate on the tune split, plus HF prefix-KV equivalence checks.

Phase 1 (equivalence, main memory r0_n256, first 20 tune queries):
  cached path : prefill M once -> DynamicCache -> tail -> greedy decode -> crop back to |M|
  full path   : one forward over M + tail + answer[:-1], no cache
  repeat      : query 0 again after other queries (checks crop really restores the cache)
  fp32 floor  : full path with an fp32 model, to calibrate bf16 noise
Phase 2 (gate): A (no memory), B (memory r{r}_n{n}), B-shuffle (same x, permuted PE)
  for n in {16,64,128,256,512} x 3 resamples, greedy decoding on every tune query.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import (  # noqa: E402
    FEATURES, MODEL_DIR, ROOT, STOP_IDS, TARGET, YARN_KWARGS, build_ids, load_splits, memory_text, parse_number, user_text,
)

MAIN_SUBSET = "r0_n256"


YARN = False      # --yarn: Qwen3 YaRN rope scaling (ccpp_common.YARN_KWARGS)
CHUNK = 16384     # --chunk: prefill M in pieces of this many tokens (memories up to this length: one forward, as before)


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype, attn_implementation="sdpa",
                                                 **(YARN_KWARGS if YARN else {}))
    return model.cuda().eval()


@torch.inference_mode()
def prefill(model, mem_ids, chunk=None):
    chunk = chunk or CHUNK
    cache = DynamicCache()
    for i in range(0, len(mem_ids), chunk):
        model(input_ids=torch.tensor([mem_ids[i:i + chunk]], device="cuda"), past_key_values=cache, use_cache=True,
              logits_to_keep=1)
    assert cache.get_seq_length() == len(mem_ids)
    return cache


@torch.inference_mode()
def greedy(model, cache, mem_len, tail, max_new, keep_logits=False):
    """Greedy decode from cache (holding exactly mem_len tokens); restores the cache afterwards."""
    assert cache.get_seq_length() == mem_len
    out = model(input_ids=torch.tensor([tail], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
    gen, logits = [], []
    for _ in range(max_new):
        lg = out.logits[0, -1].float()
        if keep_logits:
            logits.append(lg.cpu())
        nxt = int(lg.argmax())
        gen.append(nxt)
        if nxt in STOP_IDS:
            break
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache.crop(mem_len)
    assert cache.get_seq_length() == mem_len
    return gen, (torch.stack(logits) if keep_logits else None)


@torch.inference_mode()
def full_forward_logits(model, mem_ids, tail, gen):
    """Logits predicting gen[0..T-1] from a single cache-free forward over M + tail + gen[:-1]."""
    seq = mem_ids + tail + gen[:-1]
    out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=False, logits_to_keep=len(gen))
    return out.logits[0].float().cpu()


def compare(la, lb):
    """KL(softmax(la) || softmax(lb)) per position, argmax agreement, max |logit diff|."""
    pa, lpa, lpb = F.softmax(la, -1), F.log_softmax(la, -1), F.log_softmax(lb, -1)
    kl = (pa * (lpa - lpb)).sum(-1)
    return {
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "argmax_agree": float((la.argmax(-1) == lb.argmax(-1)).float().mean()),
        "max_abs_logit_diff": float((la - lb).abs().max()),
    }


def summarize_stats(per_query):
    keys = per_query[0].keys()
    return {k: {"mean": float(np.mean([d[k] for d in per_query])),
                "max": float(np.max([d[k] for d in per_query])),
                "min": float(np.min([d[k] for d in per_query]))} for k in keys}


def equivalence_phase(model, tok, rows, subsets, n_queries, max_new, out_dir):
    by_id = rows.set_index("id")
    mem_rows = by_id.loc[subsets[MAIN_SUBSET]["ids"]].reset_index()
    mem = memory_text(mem_rows)
    queries = rows[rows["split"] == "tune"].head(n_queries)

    cache = None
    cached_vs_full, records = [], []
    for qi, (_, q) in enumerate(queries.iterrows()):
        mem_ids, tail = build_ids(tok, mem, user_text(q))
        if cache is None:
            cache = prefill(model, mem_ids)
            first_mem_ids = mem_ids
        assert mem_ids == first_mem_ids
        gen, lc = greedy(model, cache, len(mem_ids), tail, max_new, keep_logits=True)
        lf = full_forward_logits(model, mem_ids, tail, gen)
        cached_vs_full.append(compare(lf, lc))
        records.append((mem_ids, tail, gen, lc, lf))

    # Re-run query 0 after the others: the cropped cache must give the same logits as the first run.
    mem_ids, tail, gen0, lc0, _ = records[0]
    gen0b, lc0b = greedy(model, cache, len(mem_ids), tail, max_new, keep_logits=True)
    repeat = {"same_tokens": gen0b == gen0, **compare(lc0, lc0b)} if len(gen0b) == len(gen0) else {"same_tokens": False}

    # Chunked prefill (pieces of 2048 tokens, as used for long memories) vs the one-forward cache above.
    del cache
    chunked = prefill(model, mem_ids, chunk=2048)
    chunked_vs_one, same = [], 0
    for mem_ids, tail, gen, lc, _ in records:
        g2, l2 = greedy(model, chunked, len(mem_ids), tail, max_new, keep_logits=True)
        if g2 == gen:
            same += 1
            chunked_vs_one.append(compare(lc, l2))
    del chunked

    report = {
        "subset": MAIN_SUBSET, "n_queries": len(records), "mem_tokens": len(records[0][0]), "yarn": YARN,
        "cached_vs_full_bf16": summarize_stats(cached_vs_full),
        "repeat_after_crop": repeat,
        "chunked2048_vs_one_forward": {"same_tokens_rate": same / len(records),
                                       **(summarize_stats(chunked_vs_one) if chunked_vs_one else {})},
    }
    return report, records


def fp32_floor(records, report):
    model32 = load_model(torch.float32)
    stats = []
    try:
        for mem_ids, tail, gen, _, lf in records:
            l32 = full_forward_logits(model32, mem_ids, tail, gen)
            stats.append(compare(l32, lf))
        report["fp32_vs_bf16_full"] = summarize_stats(stats)
    except torch.OutOfMemoryError as e:
        report["fp32_vs_bf16_full"] = f"OOM: {e}"
    del model32
    torch.cuda.empty_cache()
    return report


def gate_phase(model, tok, rows, subsets, lengths, resamples, max_new, out_dir):
    by_id = rows.set_index("id")
    tune = rows[rows["split"] == "tune"].reset_index(drop=True)
    pool_mean = float(rows[rows["split"] == "memory_pool"][TARGET].mean())

    conds = [("A", None)]
    for n in lengths:
        for r in range(resamples):
            conds += [("B", f"r{r}_n{n}"), ("B_shuffle", f"r{r}_n{n}")]
    conds.sort(key=lambda c: c[1] != MAIN_SUBSET)  # main memory first

    per_query = []
    for cond, key in conds:
        t0 = time.time()
        if key is None:
            mem, mem_pe, fallback, knn1 = None, set(), pool_mean, None
        else:
            mem_rows = by_id.loc[subsets[key]["ids"]].reset_index()
            ys = subsets[key]["shuffled_pe"] if cond == "B_shuffle" else mem_rows[TARGET].tolist()
            mem = memory_text(mem_rows, ys if cond == "B_shuffle" else None)
            mem_pe = {round(y, 2) for y in ys}
            fallback = float(np.mean(ys))
            sc = StandardScaler().fit(mem_rows[FEATURES])
            knn1 = KNeighborsRegressor(n_neighbors=1).fit(sc.transform(mem_rows[FEATURES]), ys).predict(sc.transform(tune[FEATURES]))

        cache, mem_len = None, None
        for qi, q in tune.iterrows():
            mem_ids, tail = build_ids(tok, mem, user_text(q))
            if cache is None or key is None:
                cache, mem_len = prefill(model, mem_ids), len(mem_ids)
            gen, _ = greedy(model, cache, mem_len, tail, max_new)
            text = tok.decode(gen, skip_special_tokens=True)
            pred = parse_number(text)
            per_query.append({
                "cond": cond, "subset": key or "none", "resample": -1 if key is None else int(key.split("_")[0][1:]),
                "n_rows": 0 if key is None else int(key.split("_")[1][1:]), "mem_tokens": mem_len,
                "qid": int(q["id"]), "y": float(q[TARGET]), "pred": pred, "parse_ok": pred is not None,
                "pred_filled": pred if pred is not None else fallback,
                "hit_limit": gen[-1] not in STOP_IDS, "n_gen": len(gen), "text": text,
                "copied_memory_pe": pred is not None and round(pred, 2) in mem_pe,
                "knn1": None if knn1 is None else float(knn1[qi]),
            })
        del cache
        torch.cuda.empty_cache()
        print(f"[gate] {cond:9s} {key or 'none':8s} mem_tokens={mem_len:6d} {time.time() - t0:6.1f}s", flush=True)
        pd.DataFrame(per_query).to_csv(out_dir / "gate_per_query.csv", index=False)
    return pd.DataFrame(per_query)


def paired_boot_rmse_diff(e1, e2, n_boot=2000, seed=0):
    """CI of RMSE(e1) - RMSE(e2) over resampled queries (same queries for both)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), size=(n_boot, len(e1)))
    d = np.sqrt((e1[idx] ** 2).mean(1)) - np.sqrt((e2[idx] ** 2).mean(1))
    return float(np.sqrt((e1 ** 2).mean()) - np.sqrt((e2 ** 2).mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize_gate(df, out_dir):
    rows = []
    a = df[df["cond"] == "A"].sort_values("qid")
    a_err = (a["pred_filled"] - a["y"]).to_numpy()
    for (cond, key), g in df.groupby(["cond", "subset"], sort=False):
        g = g.sort_values("qid")
        ok = g[g["parse_ok"]]
        err = (g["pred_filled"] - g["y"]).to_numpy()
        rec = {
            "cond": cond, "subset": key, "n_rows": int(g["n_rows"].iloc[0]), "resample": int(g["resample"].iloc[0]),
            "mem_tokens": int(g["mem_tokens"].iloc[0]), "n": len(g), "parse_rate": float(g["parse_ok"].mean()),
            "hit_limit_rate": float(g["hit_limit"].mean()),
            "rmse_parsed": float(np.sqrt(((ok["pred"] - ok["y"]) ** 2).mean())) if len(ok) else None,
            "mae_parsed": float((ok["pred"] - ok["y"]).abs().mean()) if len(ok) else None,
            "rmse_filled": float(np.sqrt((err ** 2).mean())), "mae_filled": float(np.abs(err).mean()),
            "copy_rate": float(ok["copied_memory_pe"].mean()) if len(ok) and cond != "A" else None,
            "corr_with_knn1": float(np.corrcoef(ok["pred"], ok["knn1"])[0, 1]) if len(ok) > 2 and cond != "A" else None,
        }
        if cond != "A":
            rec["dRMSE_vs_A"], rec["dRMSE_vs_A_lo"], rec["dRMSE_vs_A_hi"] = paired_boot_rmse_diff(err, a_err)
        if cond == "B":
            s = df[(df["cond"] == "B_shuffle") & (df["subset"] == key)].sort_values("qid")
            if len(s) == len(g):
                s_err = (s["pred_filled"] - s["y"]).to_numpy()
                rec["dRMSE_vs_shuf"], rec["dRMSE_vs_shuf_lo"], rec["dRMSE_vs_shuf_hi"] = paired_boot_rmse_diff(err, s_err)
        rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "gate_summary.csv", index=False)

    base = pd.read_csv(ROOT / "results" / "baselines_tune.csv")
    large = ROOT / "results" / "baselines_tune_large.csv"
    if large.exists():
        base = pd.concat([base, pd.read_csv(large)], ignore_index=True)
    agg = (summ[summ["cond"] != "A"].groupby(["cond", "n_rows"])
           .agg(mem_tokens=("mem_tokens", "mean"), parse_rate=("parse_rate", "mean"),
                rmse_mean=("rmse_filled", "mean"), rmse_sd=("rmse_filled", "std"),
                mae_mean=("mae_filled", "mean"), copy_rate=("copy_rate", "mean"))
           .reset_index())
    bagg = base[base["resample"] >= 0].groupby("n_rows")[["mean_rmse", "ols_rmse", "knn5_rmse", "knn1_rmse"]].mean().reset_index()
    agg = agg.merge(bagg, on="n_rows", how="left")
    agg.to_csv(out_dir / "gate_by_length.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(agg.round(3).to_string(index=False))


def main():
    global YARN, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "gate"))
    ap.add_argument("--lengths", default="16,64,128,256,512")
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--eq_queries", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--skip_fp32", action="store_true")
    ap.add_argument("--yarn", action="store_true", help="Qwen3 YaRN rope scaling (x4, up to 131072 tokens)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="prefill chunk size in tokens")
    args = ap.parse_args()
    YARN, CHUNK = args.yarn, args.chunk
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    model = load_model(torch.bfloat16)

    t0 = time.time()
    report, records = equivalence_phase(model, tok, rows, subsets, args.eq_queries, args.max_new, out_dir)
    print(json.dumps(report, indent=2), flush=True)
    (out_dir / "equivalence.json").write_text(json.dumps(report, indent=2))
    print(f"[equivalence] {time.time() - t0:.1f}s", flush=True)

    df = gate_phase(model, tok, rows, subsets, [int(x) for x in args.lengths.split(",")],
                    args.resamples, args.max_new, out_dir)
    summarize_gate(df, out_dir)

    if not args.skip_fp32:
        del model
        torch.cuda.empty_cache()
        report = fp32_floor(records, report)
        (out_dir / "equivalence.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report.get("fp32_vs_bf16_full"), indent=2), flush=True)


if __name__ == "__main__":
    main()
