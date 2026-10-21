"""Evaluate cartridges with the same HF code path the teacher uses (trained KV injected into a DynamicCache).

  select : for every checkpoint in a run dir, dev full-vocab KL(B||C) along the teacher's greedy answer path;
           writes {run_dir}/dev_selection.csv and links best_cache.pt to the lowest-KL checkpoint
  final  : on a split (default test), for every query run B greedily, then for A / B-shuffle / each cartridge:
           greedy prediction + teacher-forced full-vocab KL(B||X) along B's answer; writes per-query csv + summary
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import (  # noqa: E402
    MODEL_DIR, ROOT, STOP_IDS, TARGET, build_ids, load_splits, memory_text, parse_number, shifted_pe, user_text,
)
from teacher_gate import greedy, load_model, prefill  # noqa: E402


def cartridge_to_dynamic(path):
    """TrainableCache checkpoint -> HF DynamicCache holding the k cartridge positions (frozen first)."""
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cache = DynamicCache()
    n_layers = len(ck["trainable_keys"])
    for layer in range(n_layers):
        ks = ([ck["frozen_keys"][layer]] if ck["frozen_keys"] else []) + [ck["trainable_keys"][layer]]
        vs = ([ck["frozen_values"][layer]] if ck["frozen_values"] else []) + [ck["trainable_values"][layer]]
        k = torch.cat([t.detach() for t in ks], dim=2).to(torch.bfloat16).contiguous()
        v = torch.cat([t.detach() for t in vs], dim=2).to(torch.bfloat16).contiguous()
        cache.update(k, v, layer)
    return cache, cache.get_seq_length()


@torch.inference_mode()
def forced_logits(model, cache, mem_len, tail, gen):
    """Logits predicting gen[0..T-1] given cache + tail + gen[:-1]; restores the cache."""
    out = model(input_ids=torch.tensor([tail + gen[:-1]], device="cuda"), past_key_values=cache,
                use_cache=True, logits_to_keep=len(gen))
    cache.crop(mem_len)
    return out.logits[0].float()


def kl_stats(lp_b, lp_x):
    kl = (lp_b.exp() * (lp_b - lp_x)).sum(-1)
    return float(kl.mean()), kl.cpu().numpy(), float((lp_b.argmax(-1) == lp_x.argmax(-1)).float().mean())


def teacher_paths(model, tok, mem, users):
    cache = None
    out = []
    for u in users:
        mem_ids, tail = build_ids(tok, mem, u)
        if cache is None:
            cache, L = prefill(model, mem_ids), len(mem_ids)
        gen, logits = greedy(model, cache, L, tail, 16, keep_logits=True)
        out.append((tail, gen, F.log_softmax(logits, -1).half()))
    del cache
    torch.cuda.empty_cache()
    return out


def cmd_select(args):
    run_dir = Path(args.run_dir)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mem = Path(args.memory).read_text()
    # dev queries exactly as the teacher-data job rendered them (works for both ccpp and toy)
    users = [m[0]["content"] for m in pd.read_parquet(args.dev_parquet)["messages"]]
    if args.limit:
        users = users[: args.limit]
    model = load_model(torch.bfloat16)
    paths = teacher_paths(model, tok, mem, users)

    ckpts = [("init", run_dir / "cache_init.pt")]
    steps = sorted(run_dir.glob("cache-step*.pt"), key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    for f in steps:
        ckpts.append((f.stem, f))
    # In-loop saves are named one below their update count (see train_cartridge.py); the highest-numbered
    # file is the save_after_training checkpoint, whose name equals its update count.
    last_step = int(re.search(r"step(\d+)", steps[-1].name).group(1))
    recs = []
    for name, f in ckpts:
        cache, k = cartridge_to_dynamic(str(f))
        kls, agrees = [], []
        for tail, gen, lp_b in paths:
            lp_x = F.log_softmax(forced_logits(model, cache, k, tail, gen), -1)
            m, _, a = kl_stats(lp_b.float().cuda(), lp_x)
            kls.append(m)
            agrees.append(a)
        if name == "init":
            step = 0
        else:
            s = int(re.search(r"step(\d+)", name).group(1))
            step = s if s == last_step else s + 1
        recs.append({"ckpt": name, "updates": step, "dev_kl": float(np.mean(kls)), "dev_top1_agree": float(np.mean(agrees))})
        print(recs[-1], flush=True)
        del cache
    df = pd.DataFrame(recs)
    df.to_csv(run_dir / "dev_selection.csv", index=False)
    trained = df[df["ckpt"] != "init"]
    best = trained.loc[trained["dev_kl"].idxmin()]
    link = run_dir / "best_cache.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(f"{best['ckpt']}.pt", link)  # relative to the link's own directory
    summary = {"best_ckpt": best["ckpt"], "best_updates": int(best["updates"]), "best_dev_kl": float(best["dev_kl"]),
               "init_dev_kl": float(df.loc[df["ckpt"] == "init", "dev_kl"].iloc[0]), "n_dev": len(paths)}

    # numeric check on dev for init and best: free greedy generation, compared with the teacher and the truth
    metas = list(pd.read_parquet(args.dev_parquet)["metadata"])[: len(users)]
    for label, f in [("init", run_dir / "cache_init.pt"), ("best", run_dir / f"{best['ckpt']}.pt")]:
        cache, k = cartridge_to_dynamic(str(f))
        preds, same_text = [], []
        for u, (_, gen_b, _), m in zip(users, paths, metas):
            _, tail = build_ids(tok, None, u)
            gen, _ = greedy(model, cache, k, tail, 32)
            preds.append(parse_number(tok.decode(gen, skip_special_tokens=True)))
            same_text.append(gen == gen_b)
        y = np.array([m["y"] for m in metas], dtype=float)
        tp = np.array([np.nan if m["teacher_pred"] is None else m["teacher_pred"] for m in metas], dtype=float)
        p = np.array([np.nan if v is None else v for v in preds], dtype=float)
        ok = ~np.isnan(p)
        both = ok & ~np.isnan(tp)
        summary[f"{label}_dev_generation"] = {
            "parse_rate": float(ok.mean()),
            "rmse_vs_truth_parsed": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else None,
            "abs_diff_vs_teacher": float(np.mean(np.abs(p[both] - tp[both]))) if both.any() else None,
            "exact_same_tokens_as_teacher": float(np.mean(same_text)),
        }
        del cache
    tok_ok = ~np.isnan(tp)
    summary["teacher_dev_rmse_vs_truth"] = float(np.sqrt(np.mean((tp[tok_ok] - y[tok_ok]) ** 2))) if tok_ok.any() else None
    (run_dir / "dev_selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_final(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    by_id = rows.set_index("id")
    mem_rows = by_id.loc[subsets[args.subset]["ids"]].reset_index()
    mem = memory_text(mem_rows)
    mem_shuf = memory_text(mem_rows, subsets[args.subset]["shuffled_pe"])
    fallback = float(mem_rows[TARGET].mean())
    qs = rows[rows["split"] == args.split]
    if args.limit:
        qs = qs.head(args.limit)
    model = load_model(torch.bfloat16)

    def mem_cache(text):
        ids, _ = build_ids(tok, text, user_text(qs.iloc[0]))
        return prefill(model, ids), len(ids)

    conds = {"B": mem_cache(mem), "B_shuffle": mem_cache(mem_shuf)}
    if args.pe_shift:  # counterfactual teacher B': same memory rows with every PE shifted
        conds["B_shift"] = mem_cache(memory_text(mem_rows, shifted_pe(mem_rows, args.pe_shift, args.low_at_only)[0]))
    for spec in args.cartridges:
        name, path = spec.split("=", 1)
        conds[name] = cartridge_to_dynamic(path)

    recs = []
    for qi, (_, q) in enumerate(qs.iterrows()):
        u = user_text(q)
        _, tail = build_ids(tok, None, u)
        cache_b, Lb = conds["B"]
        gen_b, logits_b = greedy(model, cache_b, Lb, tail, args.max_new, keep_logits=True)
        lp_b = F.log_softmax(logits_b.cuda(), -1)
        names = ["A"] + list(conds)
        for name in names:
            if name == "A":
                cache, L = DynamicCache(), 0
            else:
                cache, L = conds[name]
            if name == "B":
                gen = gen_b
                kl, kl_pos, agree = 0.0, np.zeros(len(gen_b)), 1.0
            elif name == "A":
                # A never answers with a bare number in a short budget; its predictions come from
                # eval_a_long.py. Here only KL(B||A) along B's answer path is computed.
                gen = [151645]
                kl, kl_pos, agree = kl_stats(lp_b, F.log_softmax(forced_logits(model, cache, L, tail, gen_b), -1))
            else:
                gen, _ = greedy(model, cache, L, tail, args.max_new)
                kl, kl_pos, agree = kl_stats(lp_b, F.log_softmax(forced_logits(model, cache, L, tail, gen_b), -1))
            text = tok.decode(gen, skip_special_tokens=True)
            pred = parse_number(text)
            recs.append({"cond": name, "qid": int(q["id"]), "y": float(q[TARGET]), "pred": pred,
                         "parse_ok": pred is not None, "pred_filled": pred if pred is not None else fallback,
                         "hit_limit": gen[-1] not in STOP_IDS, "text": text, "cache_positions": L,
                         "kl_B_X": kl, "top1_agree_B_X": agree, "kl_by_pos": json.dumps(np.round(kl_pos, 5).tolist())})
        if qi % 100 == 0:
            print(f"[final] {qi}/{len(qs)}", flush=True)
            pd.DataFrame(recs).to_csv(out / f"final_{args.split}_per_query.csv", index=False)
    df = pd.DataFrame(recs)
    df.to_csv(out / f"final_{args.split}_per_query.csv", index=False)

    b = df[df["cond"] == "B"].set_index("qid")
    summ = []
    for name, g in df.groupby("cond", sort=False):
        g = g.set_index("qid")
        ok = g["parse_ok"]
        both = ok & b["parse_ok"].reindex(g.index)
        err = g["pred_filled"] - g["y"]
        summ.append({
            "cond": name, "n": len(g), "cache_positions": int(g["cache_positions"].iloc[0]),
            "cache_MB": float(g["cache_positions"].iloc[0] * 36 * 8 * 128 * 2 * 2 / 2 ** 20),
            "parse_rate": float(ok.mean()), "hit_limit_rate": float(g["hit_limit"].mean()),
            "rmse_filled": float(np.sqrt((err ** 2).mean())), "mae_filled": float(err.abs().mean()),
            "rmse_parsed": float(np.sqrt(((g.loc[ok, "pred"] - g.loc[ok, "y"]) ** 2).mean())) if ok.any() else None,
            "mae_parsed": float((g.loc[ok, "pred"] - g.loc[ok, "y"]).abs().mean()) if ok.any() else None,
            "abs_diff_vs_B": float((g.loc[both, "pred"] - b.loc[both[both].index, "pred"]).abs().mean()) if both.any() else None,
            "n_both_parsed": int(both.sum()),
            "kl_B_X": float(g["kl_B_X"].mean()), "top1_agree_B_X": float(g["top1_agree_B_X"].mean()),
        })
        if name == "A":
            # A's predictions are not generated here (see eval_a_long.py); only its KL is meaningful
            for key in ["parse_rate", "hit_limit_rate", "rmse_filled", "mae_filled", "rmse_parsed", "mae_parsed",
                        "abs_diff_vs_B", "n_both_parsed"]:
                summ[-1][key] = None
    s = pd.DataFrame(summ)
    s.to_csv(out / f"final_{args.split}_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("select")
    a.add_argument("--run_dir", required=True)
    a.add_argument("--memory", required=True)
    a.add_argument("--dev_parquet", required=True)
    a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("final")
    b.add_argument("--subset", default="r0_n256")
    b.add_argument("--split", default="test")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--cartridges", nargs="*", default=[], help="name=path/to/cache.pt")
    b.add_argument("--max_new", type=int, default=32)
    b.add_argument("--pe_shift", type=float, default=0.0, help="also evaluate B_shift (memory PE + this amount)")
    b.add_argument("--low_at_only", action="store_true", help="B_shift shifts only rows below the memory median AT")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    {"select": cmd_select, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
