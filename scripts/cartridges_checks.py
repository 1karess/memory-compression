"""Step 3 checks on the patched Cartridges code with Qwen3-4B (no training on our data).

  1. dtype / shape of the k=64 TrainableCache built the way train.py builds it
  2. save -> from_pretrained round trip (patch P1): frozen = 1, trainable = 63, tensors identical
  3. one Adam step at lr in {2e-3, 2e-2}: how much of the intended update survives bf16 parameters
  4. Flex (Cartridges) vs HF on identical inputs, answer positions only:
       a. cartridge = full M      vs HF over M + tail + answer
       b. cartridge = k=64 init   vs HF over the same 64 init tokens + tail + answer
Writes results/cartridges_checks.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import cartridges_compat  # noqa: E402,F401  (sets env + registers local model path)
from ccpp_common import MODEL_DIR, PROCESSED, ROOT, build_ids, load_splits, memory_text, user_text  # noqa: E402
from teacher_gate import compare, full_forward_logits, greedy, load_model, prefill  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from cartridges.cache import AttnConfig, TrainableCache  # noqa: E402
from cartridges.initialization.text import KVFromText  # noqa: E402
from cartridges.initialization.tokenization_utils import qwen_tokenize_data_into_system_prompt  # noqa: E402
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM  # noqa: E402
from cartridges.train import CacheAndModel  # noqa: E402

N_QUERIES = 8
PAD_TO = 256  # fixed packed length so flex_attention compiles once per cartridge length


def build_cache(flex, tok, text_path, max_tokens):
    cfg = flex.config
    attn_config = AttnConfig(n_layers=cfg.num_hidden_layers, n_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim)
    init = KVFromText.Config(text_source=str(text_path), max_tokens=max_tokens).instantiate()
    return init.initialize_kv_cache(tokenizer=tok, model=flex, attn_config=attn_config).cuda()


@torch.no_grad()
def flex_logits(flex, cache, tail, gen):
    """Logits predicting gen[0..T-1] given cartridge + tail + gen[:-1], packed and padded like TrainDataset."""
    ids = tail + gen[:-1]
    n = len(ids)
    assert n <= PAD_TO
    pad = PAD_TO - n
    input_ids = torch.tensor(ids + [0] * pad, device="cuda")
    seq_ids = torch.zeros(PAD_TO, dtype=torch.long, device="cuda")
    position_ids = torch.tensor(list(range(n)) + [0] * pad, device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = CacheAndModel(cache, flex)(input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids)
    cache.clear()
    T = len(gen)
    return out.logits[0, len(tail) - 1: len(tail) - 1 + T].float().cpu()


def student_loss(flex, cache, tail_gen_pairs):
    """Mean CE of the teacher-greedy answer tokens under the student (one-hot targets)."""
    loss = 0.0
    for tail, gen in tail_gen_pairs:
        ids = tail + gen[:-1]
        n = len(ids)
        pad = PAD_TO - n
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = CacheAndModel(cache, flex)(
                input_ids=torch.tensor(ids + [0] * pad, device="cuda"),
                seq_ids=torch.zeros(PAD_TO, dtype=torch.long, device="cuda"),
                position_ids=torch.tensor(list(range(n)) + [0] * pad, device="cuda"),
            )
            lp = F.log_softmax(out.logits[0, len(tail) - 1: len(tail) - 1 + len(gen)].float(), -1)
            loss = loss - lp[torch.arange(len(gen)), torch.tensor(gen, device="cuda")].mean()
        cache.clear()
    return loss / len(tail_gen_pairs)


def adam_update_test(flex, tok, text_path, lr, tail_gen_pairs, n_steps=3):
    """Take n_steps Adam steps on the bf16 cache exactly as train.py does and measure what survives rounding.

    Per step, the intended update is recomputed in fp32 from the optimizer's own moments
    (-lr * m_hat / (sqrt(v_hat) + eps)); the actual update is new - old parameter.
    """
    cache = build_cache(flex, tok, text_path, 64)
    trainable = [p for p in cache.parameters() if p.requires_grad]
    init = [p.detach().clone() for p in trainable]
    frozen_before = [p.detach().clone() for p in list(cache.frozen_keys) + list(cache.frozen_values)]
    opt = torch.optim.Adam(trainable, lr=lr)
    b1, b2 = opt.defaults["betas"]
    eps = opt.defaults["eps"]
    steps = []
    for _ in range(n_steps):
        old = [p.detach().clone() for p in trainable]
        opt.zero_grad()
        loss = student_loss(flex, cache, tail_gen_pairs)
        loss.backward()
        opt.step()
        intended, actual = [], []
        for p, o in zip(trainable, old):
            st = opt.state[p]
            t = float(st["step"])
            m_hat = st["exp_avg"].float() / (1 - b1 ** t)
            v_hat = st["exp_avg_sq"].float() / (1 - b2 ** t)
            intended.append((-lr * m_hat / (v_hat.sqrt() + eps)).flatten())
            actual.append((p.detach().float() - o.float()).flatten())
        intended, actual = torch.cat(intended), torch.cat(actual)
        nz = intended != 0
        steps.append({
            "loss_before_step": float(loss),
            "frac_elements_changed": float((actual != 0).float().mean()),
            "frac_intended_nonzero_but_rounded_to_zero": float(((actual == 0) & nz).float().sum() / nz.float().sum()),
            "rel_err_actual_vs_intended": float((actual - intended).norm() / intended.norm()),
            "max_abs_change": float(actual.abs().max()),
        })
    with torch.no_grad():
        final_loss = float(student_loss(flex, cache, tail_gen_pairs))
    total = torch.cat([(p.detach().float() - i.float()).flatten() for p, i in zip(trainable, init)])
    mags = torch.cat([i.float().abs().flatten() for i in init])
    sample = mags[torch.randperm(len(mags), device=mags.device)[:1_000_000]]
    res = {
        "lr": lr,
        "param_dtype": str(trainable[0].dtype),
        "grad_dtype": str(trainable[0].grad.dtype),
        "adam_state_dtype": str(opt.state[trainable[0]]["exp_avg"].dtype),
        "steps": steps,
        "loss_after_last_step": final_loss,
        "frac_elements_changed_vs_init": float((total != 0).float().mean()),
        "rel_change_norm_vs_init": float(total.norm() / mags.norm()),
        "kv_abs_percentiles": {p: float(torch.quantile(sample, p / 100)) for p in (50, 90, 99)},
        "model_params_have_grad": any(p.grad is not None for p in flex.parameters()),
        "frozen_requires_grad": any(p.requires_grad for p in cache.frozen_keys),
        "frozen_unchanged": all(torch.equal(p, q) for p, q in zip(list(cache.frozen_keys) + list(cache.frozen_values), frozen_before)),
    }
    del cache, opt
    torch.cuda.empty_cache()
    return res


def topk_coverage(items, ks=(10, 20, 50)):
    """Teacher (HF, full M) probability mass captured by the top-k candidates at each answer position."""
    per_pos = {}
    for it in items:
        probs = F.softmax(it["hf_full"], -1)
        srt = probs.sort(-1, descending=True).values
        for j in range(probs.shape[0]):
            per_pos.setdefault(j, []).append({k: float(srt[j, :k].sum()) for k in ks} | {"top1": float(srt[j, 0])})
    out = {}
    for j, lst in per_pos.items():
        out[f"pos{j}"] = {f"top{k}_mass_min": min(d[k] for d in lst) for k in ks}
        out[f"pos{j}"]["top20_mass_median"] = float(np.median([d[20] for d in lst]))
        out[f"pos{j}"]["top1_prob_median"] = float(np.median([d["top1"] for d in lst]))
        out[f"pos{j}"]["n"] = len(lst)
    return out


def main():
    out_path = ROOT / "results" / "cartridges_checks.json"
    report = {}
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    mem = memory_text(rows.set_index("id").loc[subsets["r0_n256"]["ids"]].reset_index())
    text_path = PROCESSED / "memory_r0_n256.txt"
    text_path.write_text(mem)
    queries = rows[rows["split"] == "tune"].head(N_QUERIES)

    # HF teacher answers + reference logits (full M and k=64-init prefixes)
    hf = load_model(torch.bfloat16)
    init64 = qwen_tokenize_data_into_system_prompt(tok, mem, 64)[0].tolist()
    items = []
    cache = None
    for _, q in queries.iterrows():
        mem_ids, tail = build_ids(tok, mem, user_text(q))
        if cache is None:
            cache = prefill(hf, mem_ids)
        gen, _ = greedy(hf, cache, len(mem_ids), tail, 16)
        items.append({"tail": tail, "gen": gen,
                      "hf_full": full_forward_logits(hf, mem_ids, tail, gen),
                      "hf_init64": full_forward_logits(hf, init64, tail, gen)})
    del cache, hf
    torch.cuda.empty_cache()
    report["teacher_topk_coverage"] = topk_coverage(items)
    report["teacher_answers"] = [tok.decode(it["gen"]) for it in items]

    flex = FlexQwen3ForCausalLM.from_pretrained(MODEL_DIR).cuda().to(torch.bfloat16)  # as in train.py
    for p in flex.parameters():
        p.requires_grad_(False)

    # 1 + 2: k=64 cache dtype and save/load round trip
    c64 = build_cache(flex, tok, text_path, 64)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "c.pt")
        c64.save(path)
        ck = torch.load(path, weights_only=False)
        loaded = TrainableCache.from_pretrained(path, device="cuda")
        report["cache_k64"] = {
            "param_dtype": str(c64.trainable_keys[0].dtype),
            "frozen_shape": list(c64.frozen_keys[0].shape), "trainable_shape": list(c64.trainable_keys[0].shape),
            "num_frozen": c64._num_frozen_tokens, "num_trainable": c64._num_trainable_tokens,
            "loaded_num_frozen": loaded._num_frozen_tokens, "loaded_num_trainable": loaded._num_trainable_tokens,
            "unpatched_would_read_num_frozen": int(ck["frozen_keys"][0].size(1)),
            "roundtrip_identical": all(torch.equal(a, b) for a, b in zip(
                list(c64.frozen_keys) + list(c64.trainable_keys) + list(c64.frozen_values) + list(c64.trainable_values),
                list(loaded.frozen_keys) + list(loaded.trainable_keys) + list(loaded.frozen_values) + list(loaded.trainable_values))),
        }
    print(json.dumps(report, indent=2), flush=True)

    # 4b: k=64 init cartridge vs HF truncated ICL; also logits from the reloaded cache must match
    s_init, s_loaded = [], []
    for it in items:
        lf = flex_logits(flex, c64, it["tail"], it["gen"])
        s_init.append(compare(it["hf_init64"], lf))
        s_loaded.append(compare(lf, flex_logits(flex, loaded.cuda(), it["tail"], it["gen"])))
    report["flex_vs_hf_init64"] = {k: float(np.mean([d[k] for d in s_init])) for k in s_init[0]}
    report["flex_cache_vs_reloaded_cache"] = {k: float(np.max([d[k] for d in s_loaded])) for k in s_loaded[0]}
    del c64, loaded
    torch.cuda.empty_cache()

    # 4a: full-M cartridge vs HF full context
    cfull = build_cache(flex, tok, text_path, None)
    s_full = [compare(it["hf_full"], flex_logits(flex, cfull, it["tail"], it["gen"])) for it in items]
    report["flex_vs_hf_fullM"] = {k: float(np.mean([d[k] for d in s_full])) for k in s_full[0]}
    report["flex_vs_hf_fullM_max_kl"] = float(np.max([d["kl_max"] for d in s_full]))
    report["fullM_cartridge_tokens"] = cfull.num_cartridge_tokens()
    del cfull
    torch.cuda.empty_cache()
    print(json.dumps(report, indent=2), flush=True)

    # 3: does one Adam step on bf16 parameters actually move them?
    pairs = [(it["tail"], it["gen"]) for it in items]
    report["adam_update"] = [adam_update_test(flex, tok, text_path, lr, pairs) for lr in (2e-3, 2e-2)]

    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
