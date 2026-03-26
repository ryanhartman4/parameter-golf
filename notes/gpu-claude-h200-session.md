# Summary: Parameter Golf H200 Session

## What We Set Up

- **Instance:** 1xH200, 150GB VRAM, CUDA 12.4, 16 CPU cores
- **Repo:** `/p_golf/parameter-golf/` — cloned and pulled latest (11 experiment variants)
- **Environment:** Python 3.11 venv, torch 2.6.0+cu124, FA3 built from source (hdim64-only)
- **Dataset:** FineWeb sp1024 — 80 train shards (~8B tokens), 1 val shard (62M tokens)

## What We Ran

- **Smoke test:** Passed. 50 steps, loss dropped 8.18 → 5.71, 624ms/step, 21GB VRAM
- **Baseline (v00):** Completed. 718 steps in 600s, sliding_bpb = 3.3660, artifact 6.0MB
- **eval_stride32:** Killed at ~25 min — was running but extremely slow (62M tokens at stride=32)

## Modifications Made to run_all.sh

1. Added `TRAIN_LOG_EVERY=100` — BPB logged every 100 steps
2. Added PATH export for venv + uv at top of script
3. Changed `torchrun` to absolute path `/p_golf/parameter-golf/.venv/bin/torchrun`

## Key Issues Identified

### 1. Step count is too low (fundamental)

- 1xH200 gets ~700 steps vs ~7100 on 8xH100 (10x fewer)
- Only sees the first 10% of training — changes that affect late training (DCA convergence, QAT warmdown, capacity saturation) are invisible
- Baseline BPB 3.37 vs competition SOTA 1.12 — huge gap

### 2. Eval scripts are way too slow

- eval_stride32 ran 25+ minutes with no sign of finishing (62M tokens, stride=32 = ~1.9M forward passes)
- eval_recurrence would be similarly slow (two-pass eval)
- Experiment plan estimated 10 min for both — actual is probably 60+ min combined
- Blocks the entire training queue since run_all.sh runs them sequentially

### 3. FA3 build was painful

- No prebuilt wheel exists for FA3 Hopper
- Full build = 451 CUDA kernel files, would take 3-4 hours
- Solved by disabling unnecessary hdim/dtype variants (451 → 4 files, built in ~5 min)
