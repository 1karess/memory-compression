"""Shared constants and prompt construction for the CCPP memory-compression experiments.

Every script that builds model inputs must go through `build_ids` so that A / B / C
see byte-identical tails and the memory M always sits in the system turn.
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (scripts/..)
MODEL_NAME = os.environ.get("MZ_MODEL", "Qwen3-4B")  # directory under models/; Qwen3-4B-Instruct-2507 for the scale-up
MODEL_DIR = ROOT / "models" / MODEL_NAME
PROCESSED = ROOT / "data" / "processed"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"

# Qwen3 special tokens
IM_START, IM_END, NEWLINE = 151644, 151645, 198
THINK_OPEN, THINK_CLOSE, DOUBLE_NEWLINE = 151667, 151668, 271
EMPTY_THINK = [THINK_OPEN, DOUBLE_NEWLINE, THINK_CLOSE, DOUBLE_NEWLINE]
# Qwen3-4B (hybrid thinking) ends the generation prompt with an empty think block under enable_thinking=False;
# the Instruct-2507 models have no thinking mode and end it right after "assistant\n".
HAS_THINKING = MODEL_NAME == "Qwen3-4B"
STOP_IDS = {IM_END, 151643}  # <|im_end|>, <|endoftext|>

INSTRUCTION = (
    "Records from a combined cycle power plant have the fields "
    "AT (ambient temperature, C), V (exhaust vacuum, cm Hg), AP (ambient pressure, mbar), "
    "RH (relative humidity, %) and PE (net hourly electrical energy output, MW).\n"
    "Predict PE for the record below. Reply with the number only, using two decimals.\n"
)

# Qwen3's recommended YaRN setting for contexts beyond the native 32k (static scaling, applied at every length).
# Passed as from_pretrained kwargs; transformers takes factor = max_position_embeddings / original (= 4).
YARN_KWARGS = {"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
               "max_position_embeddings": 131072}

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def fmt_x(row) -> str:
    return ", ".join(f"{f}={row[f]:.2f}" for f in FEATURES)


def fmt_row(row, y=None) -> str:
    y = row[TARGET] if y is None else y
    return f"{fmt_x(row)}, {TARGET}={y:.2f}"


def memory_text(rows, ys=None) -> str:
    """rows: DataFrame of memory rows; ys: optional override of PE values (for B-shuffle)."""
    if ys is None:
        return "\n".join(fmt_row(r) for _, r in rows.iterrows())
    return "\n".join(fmt_row(r, y) for (_, r), y in zip(rows.iterrows(), ys))


def shifted_pe(rows, shift, low_at_only=False):
    """Counterfactual memory labels: PE + shift for every row, or only for rows with AT below the memory's
    median AT (low_at_only). Returns (list of PE values, AT threshold or None)."""
    if not low_at_only:
        return (rows[TARGET] + shift).tolist(), None
    thr = float(rows["AT"].median())
    return (rows[TARGET] + shift * (rows["AT"] < thr)).tolist(), thr


def user_text(row) -> str:
    return INSTRUCTION + fmt_x(row)


def build_ids(tok, mem: str | None, user: str):
    """Render the whole conversation with one chat-template call, then split.

    Returns (mem_ids, tail_ids). mem_ids is [] when mem is None (condition A).
    Asserts that splitting is exact and that the no-memory rendering equals the tail.
    """
    no_mem = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if HAS_THINKING:
        assert no_mem[-4:] == EMPTY_THINK, "enable_thinking=False must append an empty think block"
    else:
        assert no_mem[-3:] == [IM_START, 77091, NEWLINE] and THINK_OPEN not in no_mem, "expected a plain assistant prompt"
    if mem is None:
        return [], list(no_mem)
    full = tok.apply_chat_template(
        [{"role": "system", "content": mem}, {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    mem_ids = tok.apply_chat_template([{"role": "system", "content": mem}], tokenize=True)
    assert full[: len(mem_ids)] == mem_ids, "memory prefix is not a prefix of the full rendering"
    tail = full[len(mem_ids):]
    assert tail == list(no_mem), "tail differs from the no-memory rendering"
    return list(mem_ids), list(tail)


def parse_number(text: str):
    m = NUM_RE.search(text)
    return float(m.group()) if m else None


def load_splits():
    import pandas as pd
    rows = pd.read_csv(PROCESSED / "rows.csv")
    subsets = json.loads((PROCESSED / "memory_subsets.json").read_text())
    large = PROCESSED / "memory_subsets_large.json"  # n >= 1024 memories (scripts/make_large_subsets.py)
    if large.exists():
        subsets.update(json.loads(large.read_text()))
    return rows, subsets
