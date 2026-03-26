# H200 GPU Instance Setup Guide

Setup instructions for running the Parameter Golf experiment harness on a 1xH200 RunPod instance. All Python operations use `uv`.

---

## 1. Clone the Repo

```bash
cd /workspace
git clone https://github.com/ryanhartman4/parameter-golf.git
cd parameter-golf
```

## 2. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

## 3. Create Environment + Install Dependencies

Single command — uv handles venv creation, Python resolution, and package install:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
  numpy tqdm torch huggingface-hub setuptools \
  "typing-extensions==4.15.0" datasets tiktoken \
  sentencepiece zstandard kernels
```

Pin Python 3.11 — flash-attn wheels and torch CUDA builds are most reliable here.

Verify core packages:

```bash
uv run python -c "import torch, sentencepiece, zstandard; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
```

## 4. Install FlashAttention 3 (Hopper)

The training script imports `flash_attn_interface.flash_attn_func` — this is FlashAttention 3, Hopper-specific. It is NOT the standard `flash-attn` pip package.

```bash
# Check if the RunPod template already has it:
uv run python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

If that fails:

```bash
# Try the pip package first (some builds include the Hopper interface):
uv pip install --python .venv/bin/python flash-attn --no-build-isolation
uv run python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"

# If still missing, build from the Hopper branch:
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn
cd /tmp/flash-attn/hopper
uv run python setup.py install
cd /workspace/parameter-golf
uv run python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

## 5. Verify GPU

```bash
uv run python -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'CUDA:   {torch.cuda.is_available()}')
print(f'GPU:    {torch.cuda.get_device_name(0)}')
print(f'VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'CUDA v: {torch.version.cuda}')
"
```

Expected: H200, 80GB or 141GB HBM3e, CUDA 12.x.

## 6. Download FineWeb Dataset

```bash
uv run python data/cached_challenge_fineweb.py --variant sp1024
```

Downloads:
- 80 training shards (~8B tokens) → `data/datasets/fineweb10B_sp1024/`
- Full validation split → same directory
- Tokenizer → `data/tokenizers/fineweb_1024_bpe.model`

Takes ~5-10 minutes. Verify:

```bash
ls data/datasets/fineweb10B_sp1024/ | wc -l   # Should show 80+ files
ls data/tokenizers/fineweb_1024_bpe.model      # Should exist
```

## 7. Smoke Test

Quick 50-step test to verify everything works end-to-end:

```bash
RUN_ID=smoke_test \
ITERATIONS=50 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=60 \
uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Should complete in ~1-2 minutes with decreasing train_loss. If this passes, the full batch will work.

## 8. Run the Experiment Batch

```bash
chmod +x experiments/run_all.sh experiments/check_progress.sh
mkdir -p experiments/results
nohup bash experiments/run_all.sh > experiments/results/run_all_master.log 2>&1 &
```

Monitor from another terminal (or SSH session):

```bash
bash experiments/check_progress.sh
```

Or watch continuously:

```bash
watch -n 30 bash experiments/check_progress.sh
```

### Expected timeline

| Phase | Time |
|-------|------|
| Baseline training | ~12 min |
| Eval tests (stride32, recurrence) | ~10 min |
| Queue A training (8 experiments) | ~96 min |
| Quant frontier sweep | ~5 min |
| Queue B training (2 experiments) | ~24 min |
| **Total** | **~2.5 hours** |

## 9. After the Run

Results are in:
- `experiments/results/FINAL_RESULTS.txt` — comparison table
- `experiments/results/*.json` — per-experiment metrics
- `experiments/results/*.log` — full training logs
- `checkpoints/baseline_raw.pt` — fp32 baseline checkpoint
- `checkpoints/baseline_compressed.ptz` — compressed baseline artifact

Push results back:

```bash
git lfs install
git add checkpoints/ experiments/results/
git commit -m "Add overnight experiment results from 1xH200 run"
git push origin main
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'flash_attn_interface'`
FlashAttention 3 (Hopper) is not installed. See Step 4. The RunPod Parameter Golf template should have it pre-installed. If using a generic template, build from the hopper branch.

### `RuntimeError: CUDA out of memory`
The model is only ~27M params — H200 has plenty of headroom. Check for other processes: `nvidia-smi`.

### zstandard not found — falls back to zlib
The script falls back gracefully, but zstd-22 gives much better compression (~15.5MB vs ~16+ MB). Fix: `uv pip install --python .venv/bin/python zstandard`.

### `torchrun` not found
Torch isn't in PATH. Use `uv run torchrun ...` which resolves through the venv automatically.

### Data download fails
Check HuggingFace Hub connectivity. Try: `HF_HUB_ENABLE_HF_TRANSFER=0 uv run python data/cached_challenge_fineweb.py --variant sp1024`.
