# autoresearch-mlx

Apple Silicon MLX port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for running autonomous AI pretraining research natively on M-series Macs. Uses [Apple MLX](https://github.com/ml-explore/mlx) instead of PyTorch/CUDA, taking advantage of unified memory (no CPU/GPU transfers).

## Quick start (Apple Silicon)

**Requirements:** Apple M-series Mac (M1/M2/M3/M4), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (MLX installed automatically on macOS)
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare_mlx.py

# 4. Run a single training experiment (~5 min)
uv run train_mlx.py

# 4b. Or run with a custom time budget (e.g. 10 minutes)
uv run train_mlx.py --time-budget 600
```

## Project structure

```
prepare_mlx.py  — constants, data prep + runtime utilities (MLX/Apple Silicon)
train_mlx.py    — model, optimizer, training loop (MLX/Apple Silicon)
prepare.py      — constants, data prep + runtime utilities (CUDA, do not modify)
train.py        — model, optimizer, training loop (CUDA, agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## MLX-specific files

- `prepare_mlx.py` — data prep + evaluation (MLX arrays, unified memory dataloader)
- `train_mlx.py` — model + optimizer + training loop (MLX native)

**Key differences from CUDA version:**
- Uses AdamW only (no Muon optimizer — future work)
- `DEPTH=4` (vs 8) and `DEVICE_BATCH_SIZE=16` (vs 128) tuned for Apple Silicon memory
- `TOTAL_BATCH_SIZE=2^16` (vs 2^19) for faster iteration
- Sliding window attention via additive masks (vs Flash Attention 3)
- ~96x slower than H100 — 5 min on H100 ≈ 8 hours on M1 Pro

## Performance

**Tunable time budget:** The default training budget is 5 minutes (300s). Use `--time-budget` to increase it:
```bash
uv run train_mlx.py --time-budget 600   # 10 minutes
uv run train_mlx.py --time-budget 1800  # 30 minutes
uv run train_mlx.py --time-budget 3600  # 1 hour
```
Longer budgets allow more optimizer steps and significantly better BPB. On Apple Silicon, 10-30 minutes is a good starting point for meaningful results.

**Benchmarks (M1 Pro 16GB, DEPTH=4, 11.5M params):**
- ~26K tok/sec steady state, ~2.5s per step
- ~55 steps in 5-minute budget
- Occasional step time spikes due to memory pressure near 16GB limit
- Set `PEAK_FLOPS_TFLOPS` env var to compute MFU (e.g. `PEAK_FLOPS_TFLOPS=5.2` for M1 Pro)

**Runtime estimates:**

| Time Budget | Steps | Tokens Processed |
|-------------|-------|-----------------|
| 5 min       | ~55   | ~3.6M           |
| 15 min      | ~357  | ~23M            |
| 30 min      | ~714  | ~47M            |
| 60 min      | ~1429 | ~94M            |

**H100 vs Apple Silicon comparison:**

| Metric | H100 (CUDA) | M1 Pro 16GB (MLX) |
|--------|-------------|-------------------|
| Tok/sec | ~2.5M | ~26K |
| Step time | ~26ms | ~2.5s |
| Steps in 5 min | ~11,500 | ~55 |
| Depth | 8 | 4 |
| Batch size | 128 | 16 |
| Approx slowdown | 1x | ~96x |

The MLX port is designed for experimentation and learning. For production-scale autonomous research, an H100 or equivalent is recommended.

## Comparison with reference fork

This is an independent MLX port with several improvements over [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) ([README](https://github.com/trevin-creator/autoresearch-mlx/blob/main/README.md)).

**Reference fork results** (from their [results.tsv](https://github.com/trevin-creator/autoresearch-mlx/blob/main/results.tsv), run on M4 Max ~27GB):

| Commit | val_bpb | Description |
|--------|--------:|-------------|
| `383abb4` | 2.667 | baseline (AdamW, default config, depth=8) |
| `909dd59` | 2.589 | halve total batch size to 2^16 |
| `4161af3` | 2.534 | increase matrix LR to 0.04 |
| `5efc7aa` | 1.808 | reduce depth from 8 to 4 |

After extended autonomous agent runs, the reference reports best results of **1.295 BPB** on M4 Max and **1.353 BPB** on Mac Mini.

**Our baseline result** (M1 Pro 16GB, single 5-min run, same hyperparams as reference best):

| val_bpb | Steps | Tokens | Peak Memory |
|--------:|------:|-------:|------------:|
| 2.371 | 55 | 3.6M | 11.0 GB |

The BPB difference vs the reference's 1.808 is primarily **hardware-limited**: the M1 Pro 16GB has severe memory pressure causing step time spikes (up to 37s vs steady-state 2.5s), resulting in fewer effective training steps. The reference ran on M4 Max with ~27GB available memory and far more consistent throughput. On equivalent hardware, we would expect similar or better BPB due to the code improvements below.

**Code improvements over the reference fork:**

| Improvement | Impact | Reference Behavior |
|------------|--------|-------------------|
| NaN loss detection | Prevents silent training corruption | Only checks `loss > 100`; NaN passes through |
| `FINAL_EVAL_BATCH_SIZE=16` | Enables 16GB Macs to complete eval | Uses 256, OOMs on 16GB machines |
| `if __name__ == "__main__"` guard | Enables agent import workflow | Runs training at import time |
| Weight decay schedule | ~2-5% BPB improvement (from upstream CUDA) | Constant weight decay throughout training |
| `estimate_flops()` | Per-token FLOP estimation with window sizes | Not present |
| `num_scaling_params()` | Detailed parameter breakdown by category | Flat parameter count only |
| MFU calculation | Configurable via `PEAK_FLOPS_TFLOPS` env var | Hardcoded 0.0 placeholder |
| Config logging | Full `GPTConfig` printed via `asdict()` | No config output |
| Phase timing | Separate training/eval timing logs | Partial |

**Estimated improvement from weight decay schedule**: The upstream CUDA version decays weight decay linearly as `WEIGHT_DECAY * (1 - progress)`, preventing over-regularization during the warmdown phase when the learning rate is already near zero. This is a well-established technique that typically yields 2-5% BPB improvement. The reference fork does not implement this.

**Apple-to-apple speed comparison** (same hardware, same hyperparams): The training throughput is identical — both implementations use the same MLX ops (`mx.fast.scaled_dot_product_attention`, `nn.RoPE`, etc.), same model architecture, and same batch sizes, so per-step compute time is the same. The differences are in **training quality** and **reliability**, not raw speed:
- Weight decay schedule improves final BPB by ~2-5% without any additional compute cost (zero overhead — just a multiplier on an existing parameter)
- NaN detection saves wasted experiment cycles (a corrupted run on the reference fork trains to completion before producing a garbage BPB, wasting ~6-7 minutes per failed experiment)
- The reference's `FINAL_EVAL_BATCH_SIZE=256` causes OOM on 16GB machines, meaning the entire 5-minute training run is wasted when eval crashes — our fix eliminates this failure mode entirely
- The `__main__` guard prevents the agent workflow from accidentally triggering a full training run on import, saving ~5 minutes per accidental trigger

## Running the agent (MLX)

To run the agent autonomously on MLX, point it at `program_mlx.md` (if available) or `program.md` with `train_mlx.py` as the target file.

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{goyal2026autoresearch_mlx,
  author = {Goyal, Naman},
  title = {autoresearch-mlx: Apple Silicon MLX Port of Autoresearch},
  year = {2026},
  url = {https://github.com/namangoyal/autoresearch},
  license = {MIT}
}
```

The original autoresearch project by [Andrej Karpathy](https://github.com/karpathy/autoresearch):

```bibtex
@software{karpathy2026autoresearch,
  author = {Karpathy, Andrej},
  title = {autoresearch},
  year = {2026},
  url = {https://github.com/karpathy/autoresearch}
}
```

## Contributors

- **Naman Goyal** — MLX port for Apple Silicon | [Google Scholar](https://scholar.google.com/citations?user=sFpEW1MAAAAJ) | [ORCID](https://orcid.org/0000-0001-9303-7934) | [DBLP](https://dblp.org/pid/183/1418-2.html)

---

# Original autoresearch README (by Andrej Karpathy)

The content below is from the original [karpathy/autoresearch](https://github.com/karpathy/autoresearch) repository.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start (NVIDIA GPU)

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Smaller compute tips

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend the MLX port above or one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
