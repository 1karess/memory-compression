"""Make Cartridges' name-keyed lookup tables accept our local Qwen3-4B directory.

Cartridges looks up per-model helpers by tokenizer.name_or_path, which is the local path
when the model is loaded from models/Qwen3-4B. Register that path as an alias of
"Qwen/Qwen3-4b" (the spelling Cartridges' case-sensitive MODELS_WITH_THINKING expects).
Import this module before building any Cartridges dataset or initializer.
"""
import os

from ccpp_common import MODEL_DIR, ROOT

os.environ.setdefault("CARTRIDGES_DIR", str(ROOT / "third_party" / "cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(ROOT / "outputs" / "cartridges"))
os.environ.setdefault("WANDB_MODE", "offline")

from cartridges import datasets as _datasets  # noqa: E402
from cartridges.initialization import tokenization_utils as _tok_utils  # noqa: E402
from cartridges.utils import thinking as _thinking  # noqa: E402

CANONICAL = "Qwen/Qwen3-4b"


def register(path=MODEL_DIR):
    for name in {str(path), str(path).rstrip("/") + "/"}:
        _datasets.MODEL_TO_MESSAGE_CONVERTER[name.lower()] = _datasets.MODEL_TO_MESSAGE_CONVERTER[CANONICAL.lower()]
        _tok_utils.MODEL_TO_SYSTEM_PROMPT_TOKENIZER[name.lower()] = _tok_utils.MODEL_TO_SYSTEM_PROMPT_TOKENIZER[CANONICAL.lower()]
        _thinking.MODEL_TO_THINKING_OVERRIDES[name.lower()] = _thinking.MODEL_TO_THINKING_OVERRIDES[CANONICAL.lower()]
        _tok_utils.MODELS_WITH_THINKING.add(name)


register()
