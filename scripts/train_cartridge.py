"""Train a k-position cartridge Z with the (patched) Cartridges trainer on our teacher data.

Saves every --save_every optimizer steps (all checkpoints kept) so that eval_cartridge.py can pick
the best one by dev full-vocab KL(B||C). Also saves Z_init (the untrained initialization).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cartridges_compat  # noqa: E402,F401
from ccpp_common import MODEL_DIR, ROOT  # noqa: E402

from cartridges.datasets import DataSource, TrainDataset  # noqa: E402
from cartridges.initialization import KVFromText  # noqa: E402
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig  # noqa: E402
from cartridges.train import TrainConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="teacher parquet")
    ap.add_argument("--memory", required=True, help="text file Z is initialized from (KVFromText source)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N teacher conversations (0 = all)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8, help="packed sequences per optimizer step")
    ap.add_argument("--packed", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=0, help="0 = once per epoch")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:  # distillation-size sweep: train on the first N teacher conversations only
        from cartridges.structs import read_conversations, write_conversations
        convos = read_conversations(args.data)
        assert len(convos) >= args.limit, f"{args.data} has {len(convos)} < {args.limit} conversations"
        args.data = str(run_dir / f"distill_first{args.limit}.parquet")
        write_conversations(convos[: args.limit], args.data)

    # steps per epoch = ceil(#packed sequences / batch); count packed sequences the way TrainDataset will
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ds_cfg = TrainDataset.Config(
        data_sources=[DataSource(path=args.data, type="local")],
        top_k_logits=20, packed_seq_length=args.packed, packing_mode="pad",
    )
    n_packed = len(ds_cfg.instantiate(tokenizer=tok, seed=args.seed))
    steps_per_epoch = max(1, n_packed // args.batch)
    save_every = args.save_every or steps_per_epoch

    config = TrainConfig(
        name=run_dir.name,
        output_dir=str(run_dir.parent),
        run_dir=str(run_dir),
        model=HFModelConfig(pretrained_model_name_or_path=str(MODEL_DIR), model_cls=FlexQwen3ForCausalLM),
        kv_cache_initializer=KVFromText.Config(text_source=args.memory, max_tokens=args.k),
        lr=args.lr,
        epochs=args.epochs,
        global_batch_size=args.batch,
        dataset=ds_cfg,
        loss_evals=[],
        generate_evals=[],
        wandb=None,
        save_every_n_steps=save_every,
        save_after_training=True,
        keep_last_n_saved=10 ** 6,
        save_to_wandb=False,
        max_optimizer_steps=args.max_steps,
        seed=args.seed,
    )
    # keep Z_init: wrap the initializer so the exact cache the trainer starts from is saved
    orig_init = KVFromText.initialize_kv_cache

    def init_and_save(self, *a, **kw):
        cache = orig_init(self, *a, **kw)
        cache.save(str(run_dir / "cache_init.pt"))
        return cache

    KVFromText.initialize_kv_cache = init_and_save

    # NOTE: train.py's in-loop save runs after optimizer.step() but before the step counter
    # increments, so in-loop cache-step{M}.pt holds M+1 updates; the final save_after_training
    # file cache-step{N}.pt (highest number) holds exactly N updates.
    meta = {**vars(args), "n_packed_sequences": n_packed, "steps_per_epoch": steps_per_epoch, "save_every": save_every}
    (run_dir / "train_args.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    train(config)
    (run_dir / "TRAIN_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
