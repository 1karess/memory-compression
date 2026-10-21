"""Condition A (no memory) with a long answer budget.

With a 16-token budget Qwen3-4B never answers A with a bare number: it starts explaining
("To predict the net hourly electrical energy output ...") and is cut off. Here A may write up to
--max_new tokens; we take the last number in the output. Batched greedy generation (no memory, so
left padding is safe). Writes {out}/A_long_{split}.csv and prints RMSE / MAE / parse rates.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import MODEL_DIR, NUM_RE, TARGET, build_ids, load_splits, user_text  # noqa: E402
from teacher_gate import load_model  # noqa: E402

BARE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune")
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    rows, _ = load_splits()
    qs = rows[rows["split"] == args.split]
    if args.limit:
        qs = qs.head(args.limit)
    pool_mean = float(rows[rows["split"] == "memory_pool"][TARGET].mean())
    model = load_model(torch.bfloat16)

    recs = []
    items = list(qs.iterrows())
    for i in range(0, len(items), args.batch):
        chunk = items[i: i + args.batch]
        tails = [build_ids(tok, None, user_text(q))[1] for _, q in chunk]
        L = max(map(len, tails))
        ids = torch.tensor([[tok.pad_token_id] * (L - len(t)) + t for t in tails], device="cuda")
        mask = torch.tensor([[0] * (L - len(t)) + [1] * len(t) for t in tails], device="cuda")
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for (_, q), g in zip(chunk, gen[:, L:].tolist()):
            ended = 151645 in g
            text = tok.decode(g, skip_special_tokens=True)
            nums = NUM_RE.findall(text)
            pred = float(nums[-1]) if nums else None
            recs.append({"qid": int(q["id"]), "y": float(q[TARGET]), "pred": pred, "parse_ok": pred is not None,
                         "bare_number": bool(BARE.match(text)), "ended": ended,
                         "n_gen": (g.index(151645) + 1) if ended else len(g),
                         "pred_filled": pred if pred is not None else pool_mean, "text": text})
        print(f"[A_long] {min(i + args.batch, len(items))}/{len(items)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(out / f"A_long_{args.split}.csv", index=False)
    ok = df["parse_ok"] & df["ended"]
    err = df["pred_filled"] - df["y"]
    print({
        "n": len(df), "ended_rate": float(df["ended"].mean()), "parse_rate(ended & number)": float(ok.mean()),
        "bare_number_rate": float(df["bare_number"].mean()),
        "rmse_parsed": float(np.sqrt(((df.loc[ok, "pred"] - df.loc[ok, "y"]) ** 2).mean())) if ok.any() else None,
        "mae_parsed": float((df.loc[ok, "pred"] - df.loc[ok, "y"]).abs().mean()) if ok.any() else None,
        "rmse_filled": float(np.sqrt((err ** 2).mean())), "mean_n_gen": float(df["n_gen"].mean()),
    })


if __name__ == "__main__":
    main()
