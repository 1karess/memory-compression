"""Token-level audit: HF chat template (enable_thinking=False) vs the Cartridges Qwen3 code paths.

Checks
  1. name-based lookups in Cartridges for the model names we might use
  2. Cartridges system-prompt tokenizer == our memory prefix (full and k=64 truncation)
  3. Cartridges message converter (pre-tokenized and retokenize paths) vs HF rendering of user+assistant
  4. Cartridges GenerateEvalDataset-style prompt with/without the enable_thinking kwarg
  5. digit tokenization of PE-like numbers
Writes results/token_audit.json. Needs CARTRIDGES_DIR / CARTRIDGES_OUTPUT_DIR set.
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from ccpp_common import (  # noqa: E402
    EMPTY_THINK, IM_END, MODEL_DIR, ROOT, build_ids, load_splits, memory_text, user_text,
)

from cartridges.clients.base import FlatTopLogprobs  # noqa: E402
from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER, qwen_messages_to_element  # noqa: E402
from cartridges.initialization.tokenization_utils import (  # noqa: E402
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER, MODELS_WITH_THINKING, qwen_tokenize_data_into_system_prompt,
)
from cartridges.structs import Conversation  # noqa: E402
from cartridges.utils.thinking import MODEL_TO_THINKING_OVERRIDES  # noqa: E402


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows, subsets = load_splits()
    by_id = rows.set_index("id")
    mem = memory_text(by_id.loc[subsets["r0_n256"]["ids"]].reset_index())
    q = rows[rows["split"] == "tune"].iloc[0]
    report = {}

    # 1. lookups
    names = {"local_path": tok.name_or_path, "official": "Qwen/Qwen3-4B", "cartridges_style": "Qwen/Qwen3-4b"}
    report["name_lookups"] = {
        label: {
            "name": n,
            "MODEL_TO_MESSAGE_CONVERTER[name.lower()]": n.lower() in MODEL_TO_MESSAGE_CONVERTER,
            "MODEL_TO_SYSTEM_PROMPT_TOKENIZER[name.lower()]": n.lower() in MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
            "MODEL_TO_THINKING_OVERRIDES[name.lower()]": n.lower() in MODEL_TO_THINKING_OVERRIDES,
            "name in MODELS_WITH_THINKING (case-sensitive)": n in MODELS_WITH_THINKING,
        }
        for label, n in names.items()
    }

    # 2. system prompt / memory prefix
    mem_ids, tail = build_ids(tok, mem, user_text(q))
    cart_full = qwen_tokenize_data_into_system_prompt(tok, mem, None)[0].tolist()
    cart_64 = qwen_tokenize_data_into_system_prompt(tok, mem, 64)[0].tolist()
    report["system_prompt"] = {
        "mem_tokens": len(mem_ids),
        "cartridges_full_equals_ours": cart_full == mem_ids,
        "first_diff": first_diff(cart_full, mem_ids),
        "k64_init_len": len(cart_64),
        "k64_init_equals_first_62_of_M_plus_end": cart_64 == mem_ids[:62] + [IM_END, 198],
        "k64_init_equals_first_64_of_M": cart_64 == mem_ids[:64],
        "k64_init_tail_decoded": tok.decode(cart_64[-8:]),
    }

    # 3. message converter vs HF rendering (no system; the system part is the cartridge)
    answer = "463.26"
    ans_ids = tok.encode(answer, add_special_tokens=False)
    user_ids = tok.encode(user_text(q), add_special_tokens=False)
    hf = tok.apply_chat_template(
        [{"role": "user", "content": user_text(q)}, {"role": "assistant", "content": answer}],
        tokenize=True, enable_thinking=False,
    )
    hf_gen_prompt = tail
    target_seq = hf_gen_prompt + ans_ids + [IM_END]

    def convert(assistant_ids, retokenize=False):
        # The pre-tokenized path needs top_logprobs on at least one message; a dummy 1-candidate view suffices.
        n = len(assistant_ids) if assistant_ids is not None else 0
        dummy = FlatTopLogprobs(
            token_idx=np.arange(n, dtype=np.int64), token_id=np.array(assistant_ids or [], dtype=np.int64),
            logprobs=np.zeros(n, dtype=np.float32), shape=(n, 1),
        ) if n else None
        msgs = [
            Conversation.Message(content=user_text(q), role="user", token_ids=user_ids),
            Conversation.Message(content=answer, role="assistant", token_ids=assistant_ids, top_logprobs=dummy),
        ]
        return qwen_messages_to_element(msgs, retokenize=retokenize, tokenizer=tok).input_ids.tolist()

    plain = convert(ans_ids + [IM_END])
    with_think = convert(EMPTY_THINK + ans_ids + [IM_END])
    retok = convert(None, retokenize=True)
    report["message_converter"] = {
        "target(HF gen prompt + answer + im_end)": tok.decode(target_seq[-16:]),
        "hf_full_template_tail": tok.decode(hf[-16:]),
        "hf_full_template_equals_target(+newline)": hf == target_seq + [198],
        "cartridges_plain_equals_target": plain == target_seq,
        "cartridges_plain_first_diff": first_diff(plain, target_seq),
        "cartridges_plain_tail": tok.decode(plain[-12:]),
        "cartridges_with_empty_think_equals_target": with_think == target_seq,
        "cartridges_retokenize_equals_target": retok == target_seq,
        "cartridges_retokenize_tail": tok.decode(retok[-12:]),
    }

    # 4. GenerateEvalDataset-style prompt: kwarg passed only if name in MODELS_WITH_THINKING
    msgs = [{"role": "user", "content": user_text(q)}]
    with_kw = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    without_kw = tok.apply_chat_template(msgs, add_generation_prompt=True)
    report["generate_eval_prompt"] = {
        "with_enable_thinking_False_ends_with_empty_think": with_kw[-4:] == EMPTY_THINK,
        "without_kwarg_ends_with_empty_think": without_kw[-4:] == EMPTY_THINK,
        "without_kwarg_tail": tok.decode(without_kw[-6:]),
        "note": "without the kwarg Qwen3 defaults to thinking mode: no empty think block, the model may emit <think>...",
    }

    # 5. digits
    report["digit_tokenization"] = {s: [tok.decode([t]) for t in tok.encode(s, add_special_tokens=False)]
                                    for s in ["463.26", "PE=463.26", "1024.07", " 463.26"]}

    out = ROOT / "results" / "token_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
