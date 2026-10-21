# Memory compression on UCI CCPP (Cartridges + Qwen3-4B)

Can a frozen LLM's long in-context memory be compressed into a small trainable KV cache ("cartridge") by distillation?

- **A** no memory: only the question.
- **B** full memory (teacher): 256 rows of CCPP data (~11k tokens) in the system prompt, then the question.
- **C / Z** compressed memory (student): a 64-position trainable KV cache, distilled from B's top-20 log-probs.

Task: predict PE (power output) from AT, V, AP, RH. Model: frozen `Qwen/Qwen3-4B` (thinking off).
Method: [Cartridges](https://github.com/HazyResearch/cartridges) at commit `ef34ba9`, with the patches in
`patches/cartridges.diff`: P1 cache reload size fix, P2 stop at max optimizer steps, P4 fp32 master cache params
(`MZ_FP32_CACHE=1`).

**Progress log** (every instruction, action and code change): [`docs/experiment_log.md`](docs/experiment_log.md).
**Report** (Chinese, Word): `docs/记忆压缩实验_阶段0_结果报告.docx`.

## Data splits (rows, no input overlap across splits)
memory_pool 1024 · distill 1024 · dev 256 · test 2000 · tune 256 · reserve 5008 (seed 0, `scripts/prepare_data.py`).
Large memories (1024–2560 rows) for the scale-up come from memory_pool + reserve (`scripts/make_large_subsets.py`).

## Pipeline (`jobs/`, Slurm, all on compute nodes)
| Job | What |
|---|---|
| 00_setup | venv, packages, model, data download |
| 01_data_audit, 01b_token_audit | splits, memory subsets, classic baselines; token-level prompt checks |
| 02_teacher_gate | does B use the memory? (A / B / B-shuffle over memory lengths) + prefix-KV equivalence |
| 03_cartridges_checks | patched Cartridges vs HF outputs, save/load round trip |
| 05_main | teacher data → train Z (lr 2e-2, 640 updates, seeds 42/43/44) → pick checkpoint by dev KL(B‖C) → test table |
| 06–09 | report, bootstrap stats, figures, pseudo-label regressor control |
| 10–14 | controls: shuffled teacher, unrelated init, 256/64 distill rows, synthetic inputs, PE+10 global / low-AT region |
| 15–18 | validation: control seeds, other memories r1/r2, cartridge size k = 16/64/256 |
| 19–20 | scale-up to 512-row memories |
| 21–22 | long-memory (up to 2560 rows, ~110k tokens) teacher gates: YaRN, Qwen3-4B-Instruct-2507 (`MZ_MODEL`) |
| 96 | Word report |

Shared prompt and data code: `scripts/ccpp_common.py`. Summaries: `results/*.md`, `results/*.json`.

## Main results so far (test, 2000 queries, RMSE in MW)
| | B (teacher) | Z (k=64, 3 seeds) | recovery of B's gain over no memory |
|---|---|---|---|
| r0, 256 rows | 9.79 | 10.01 ± 0.19 | 97% |
| r1 / r2, 256 rows | 8.56 / 8.48 | 7.89 / 8.24 | 108% / 103% |
| r0 / r1, 512 rows | 6.64 / 6.81 | 6.49 / 6.49 | 101% / 103% |

Z faithfully follows the teacher's memory-dependent behavior (including region-only counterfactual shifts); this is
behavior distillation, and an OLS fitted on the same teacher answers is not worse in RMSE. 10× memories (2560 rows) do
not pass the teacher gate with 4B models yet (see `docs/experiment_log.md`).

## Reproduce
`sbatch jobs/00_setup.sh`, then `git -C third_party/cartridges apply ../../patches/cartridges.diff` on a Cartridges
checkout at `ef34ba9`, then the jobs in order, submitted from the project root (`jobs/submit.sh` reads partition names
from an untracked `jobs/cluster.local.sh`). Never run code on the login node.
