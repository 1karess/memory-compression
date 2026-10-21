"""Check patch P4 (fp32 master cache parameters, MZ_FP32_CACHE=1) before using it for training.

  1. parameters are fp32; the forward pass gives identical logits to the original bf16 cache
     (the bf16 -> fp32 -> bf16 round trip is exact at initialization)
  2. three Adam steps at lr 2e-3 and 2e-2: gradients reach the fp32 parameters, nothing is rounded away,
     loss decreases, frozen first position and model weights untouched
Writes results/p4_check.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ["MZ_FP32_CACHE"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
import cartridges.cache as cache_mod  # noqa: E402
import cartridges_checks as cc  # noqa: E402
import torch  # noqa: E402
from ccpp_common import MODEL_DIR, PROCESSED, ROOT, build_ids, load_splits, memory_text, user_text  # noqa: E402
from teacher_gate import greedy, load_model, prefill  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    assert cache_mod._FP32_CACHE
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    mem = memory_text(rows.set_index("id").loc[subsets["r0_n256"]["ids"]].reset_index())
    text_path = PROCESSED / "memory_r0_n256.txt"
    text_path.write_text(mem)
    queries = rows[rows["split"] == "tune"].head(cc.N_QUERIES)

    hf = load_model(torch.bfloat16)
    pairs, cache = [], None
    for _, q in queries.iterrows():
        mem_ids, tail = build_ids(tok, mem, user_text(q))
        if cache is None:
            cache = prefill(hf, mem_ids)
        gen, _ = greedy(hf, cache, len(mem_ids), tail, 16)
        pairs.append((tail, gen))
    del cache, hf
    torch.cuda.empty_cache()

    flex = cc.FlexQwen3ForCausalLM.from_pretrained(MODEL_DIR).cuda().to(torch.bfloat16)
    for p in flex.parameters():
        p.requires_grad_(False)

    report = {}
    c32 = cc.build_cache(flex, tok, text_path, 64)
    cache_mod._FP32_CACHE = False
    c16 = cc.build_cache(flex, tok, text_path, 64)
    cache_mod._FP32_CACHE = True
    diffs = [cc.compare(cc.flex_logits(flex, c16, t, g), cc.flex_logits(flex, c32, t, g)) for t, g in pairs]
    report["forward"] = {
        "fp32_param_dtype": str(c32.trainable_keys[0].dtype), "bf16_param_dtype": str(c16.trainable_keys[0].dtype),
        "max_abs_logit_diff": max(d["max_abs_logit_diff"] for d in diffs), "min_argmax_agree": min(d["argmax_agree"] for d in diffs),
    }
    del c32, c16
    torch.cuda.empty_cache()
    report["adam_update_fp32"] = [cc.adam_update_test(flex, tok, text_path, lr, pairs) for lr in (2e-3, 2e-2)]
    print(json.dumps(report, indent=2))
    # hard requirements before P4 is used for training (the main job stops here otherwise)
    assert report["forward"]["fp32_param_dtype"] == "torch.float32"
    assert report["forward"]["max_abs_logit_diff"] == 0.0, "fp32 master cache changes the forward pass"
    for r in report["adam_update_fp32"]:
        assert r["frozen_unchanged"] and not r["model_params_have_grad"] and not r["frozen_requires_grad"]
        assert all(s["frac_intended_nonzero_but_rounded_to_zero"] < 1e-3 for s in r["steps"]), "updates still rounded away"
        assert r["loss_after_last_step"] < r["steps"][0]["loss_before_step"], "loss did not decrease"
    (ROOT / "results" / "p4_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
