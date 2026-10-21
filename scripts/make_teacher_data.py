"""Teacher data for cartridge training: B (frozen Qwen3-4B + full memory M) answers each query greedily;
we keep the top-K log-probs at every answer position and write Cartridges-format conversations.

Tasks
  ccpp : memory = a CCPP memory subset, queries = rows of a split
  toy  : y = 3x + 7 (pipeline smoke test), memory / queries generated here with a fixed seed

Outputs in --out_dir:
  {name}.parquet          Cartridges Conversations (user turn = instruction + x, assistant = <empty think> + answer)
  {name}_meta.csv         qid, y, teacher text / prediction, per-query coverage stats
  {name}_full_logprobs.pt (with --save_full) full-vocab teacher log-probs per answer position, fp16
  memory.txt              the memory text the teacher saw (used as KVFromText source)
Every element is checked to tokenize, through Cartridges' converter, to exactly the tokens the teacher saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import cartridges_compat  # noqa: E402,F401
from ccpp_common import (  # noqa: E402
    EMPTY_THINK, FEATURES, IM_END, IM_START, MODEL_DIR, ROOT, TARGET, build_ids, load_splits, memory_text, parse_number,
    shifted_pe, user_text,
)
from teacher_gate import greedy, load_model, prefill  # noqa: E402

from cartridges.clients.base import FlatTopLogprobs  # noqa: E402
from cartridges.datasets import qwen_messages_to_element  # noqa: E402
from cartridges.structs import Conversation, write_conversations  # noqa: E402

TOY_INSTRUCTION = "Every record follows the same rule linking x to y. Predict y for the record below. Reply with the number only.\n"


def toy_task(n_mem, n_query, seed):
    rng = np.random.default_rng(seed)
    xs = rng.choice(np.arange(0, 1000), size=n_mem + n_query, replace=False)
    mem = "\n".join(f"x={x}, y={3 * x + 7}" for x in xs[:n_mem])
    queries = [{"qid": int(x), "y": float(3 * x + 7), "user": TOY_INSTRUCTION + f"x={x}"} for x in xs[n_mem:]]
    return mem, queries


def ccpp_task(subset, split, limit, shuffle_pe=False, n_synth=1024, pe_shift=0.0, low_at_only=False):
    """split "synth": n_synth query-independent inputs, each feature drawn uniformly (independently) over the
    memory pool's range, rounded to 2 decimals; they have no true PE (y = nan) and negative qids."""
    rows, subsets = load_splits()
    mem_rows = rows.set_index("id").loc[subsets[subset]["ids"]].reset_index()
    assert not (shuffle_pe and pe_shift), "use either --shuffle_pe or --pe_shift"
    if shuffle_pe:
        mem = memory_text(mem_rows, subsets[subset]["shuffled_pe"])
    elif pe_shift:
        mem = memory_text(mem_rows, shifted_pe(mem_rows, pe_shift, low_at_only)[0])  # counterfactual memory M'
    else:
        mem = memory_text(mem_rows)
    if split == "synth":
        pool = rows[rows["split"] == "memory_pool"]
        rng = np.random.default_rng(1)
        X = {f: rng.uniform(pool[f].min(), pool[f].max(), n_synth).round(2) for f in FEATURES}
        q = pd.DataFrame({"id": -1 - np.arange(n_synth), **X, TARGET: np.nan})
    else:
        q = rows[rows["split"] == split]
    if limit:
        q = q.head(limit)
    queries = [{"qid": int(r["id"]), "y": float(r[TARGET]), "user": user_text(r), **{f: float(r[f]) for f in FEATURES}}
               for _, r in q.iterrows()]
    return mem, queries


def to_conversation(tail, gen, top_ids, top_lps, user_content, answer_text, meta):
    """Build a Conversation whose Cartridges tokenization equals tail + gen."""
    # tail = <im_start> user \n {content} <im_end> \n <im_start> assistant \n <empty think>
    assert tail[:3] == [IM_START, 872, 198]
    end = tail.index(IM_END)
    user_ids = tail[3:end]
    assert tail[end:] == [IM_END, 198, IM_START, 77091, 198] + EMPTY_THINK
    T, K = top_ids.shape
    assistant_ids = EMPTY_THINK + gen
    flat = FlatTopLogprobs(
        token_idx=np.repeat(np.arange(len(EMPTY_THINK), len(EMPTY_THINK) + T), K).astype(np.int64),
        token_id=top_ids.reshape(-1).astype(np.int64),
        logprobs=top_lps.reshape(-1).astype(np.float32),
        shape=(len(assistant_ids), K),
    )
    convo = Conversation(
        messages=[
            Conversation.Message(content=user_content, role="user", token_ids=user_ids),
            Conversation.Message(content=answer_text, role="assistant", token_ids=assistant_ids, top_logprobs=flat),
        ],
        system_prompt="",
        metadata=meta,
    )
    el = qwen_messages_to_element(convo.messages)
    assert el.input_ids.tolist() == tail + gen, "Cartridges tokenization differs from what the teacher saw"
    idx = el.topk_token_idxs.view(T, K)[:, 0].tolist()
    assert idx == list(range(len(tail), len(tail) + T)), "top-k targets are not aligned to the answer tokens"
    return convo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ccpp", "toy"], required=True)
    ap.add_argument("--subset", default="r0_n256")
    ap.add_argument("--splits", default="distill", help="comma-separated CCPP splits (or synth)")
    ap.add_argument("--shuffle_pe", action="store_true", help="teacher = B-shuffle (memory with permuted PE)")
    ap.add_argument("--n_synth", type=int, default=1024)
    ap.add_argument("--pe_shift", type=float, default=0.0, help="teacher = B with every PE in M shifted by this amount")
    ap.add_argument("--low_at_only", action="store_true", help="shift only memory rows with AT below the memory median")
    ap.add_argument("--limit", type=int, default=0, help="first N queries per split (0 = all)")
    ap.add_argument("--toy_mem", type=int, default=64)
    ap.add_argument("--toy_queries", type=int, default=250, help="first 200 -> train, rest -> dev")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--save_full", default="", help="comma-separated split names whose full-vocab log-probs are saved")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "toy":
        mem, qs = toy_task(args.toy_mem, args.toy_queries, seed=0)
        jobs = {"train": qs[:200], "dev": qs[200:]}
    else:
        jobs = {}
        for split in args.splits.split(","):
            mem, jobs[split] = ccpp_task(args.subset, split, args.limit, args.shuffle_pe, args.n_synth, args.pe_shift,
                                          args.low_at_only)
    (out / "memory.txt").write_text(mem)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(torch.bfloat16)
    save_full = set(filter(None, args.save_full.split(",")))
    summary = {"task": args.task, "subset": args.subset if args.task == "ccpp" else "toy", "topk": args.topk,
               "shuffle_pe": args.shuffle_pe, "pe_shift": args.pe_shift, "low_at_only": args.low_at_only}

    cache, mem_len = None, None
    for name, queries in jobs.items():
        convos, metas, full = [], [], []
        for q in queries:
            mem_ids, tail = build_ids(tok, mem, q["user"])
            if cache is None:
                cache, mem_len = prefill(model, mem_ids), len(mem_ids)
            assert len(mem_ids) == mem_len
            gen, logits = greedy(model, cache, mem_len, tail, args.max_new, keep_logits=True)
            lps = F.log_softmax(logits, -1)
            top = lps.topk(args.topk, -1)
            mass = top.values.exp().sum(-1).numpy()
            text = tok.decode(gen, skip_special_tokens=True)
            pred = parse_number(text)
            meta = {"qid": q["qid"], "y": q["y"], **{f: q[f] for f in FEATURES if f in q},
                    "teacher_text": text, "teacher_pred": pred,
                    "n_gen": len(gen), "ended": gen[-1] == IM_END,
                    "topk_mass_min": float(mass.min()), "topk_mass_per_pos": json.dumps(mass.round(5).tolist())}
            convos.append(to_conversation(tail, gen, top.indices.numpy(), top.values.numpy(), q["user"], text,
                                          {k: v for k, v in meta.items() if k != "topk_mass_per_pos"}))
            metas.append(meta)
            if name in save_full:
                full.append(lps.half())
        write_conversations(convos, str(out / f"{name}.parquet"))
        mdf = pd.DataFrame(metas)
        mdf.to_csv(out / f"{name}_meta.csv", index=False)
        if full:
            torch.save({"qids": [m["qid"] for m in metas], "logprobs": full}, out / f"{name}_full_logprobs.pt")
        per_pos = [json.loads(s) for s in mdf["topk_mass_per_pos"]]
        ok = mdf["teacher_pred"].notna()
        summary[name] = {
            "n": len(mdf), "mem_tokens": mem_len, "parse_rate": float(ok.mean()), "ended_rate": float(mdf["ended"].mean()),
            "teacher_rmse": float(np.sqrt(((mdf.loc[ok, "teacher_pred"] - mdf.loc[ok, "y"]) ** 2).mean())),
            "topk_mass_min": float(mdf["topk_mass_min"].min()),
            "topk_mass_median_of_min": float(mdf["topk_mass_min"].median()),
            "frac_queries_any_pos_below_0.99": float((mdf["topk_mass_min"] < 0.99).mean()),
            "topk_mass_min_by_pos": [float(min(p[j] for p in per_pos if len(p) > j)) for j in range(max(map(len, per_pos)))],
        }
        print(json.dumps(summary[name], indent=2), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
