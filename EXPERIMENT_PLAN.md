# Parameter Golf — Experiment Plan

**Goal:** Beat the current SOTA of 1.1228 BPB (signalrush, 11L EMA + GPTQ-lite + warmdown3500)

**Hardware:** 1x H200 GPU (RunPod). All experiments run here overnight via a single `run_all.sh` script.

**Baseline:** Root `train_gpt.py` with 4 bug fixes + 2 efficiency fixes applied 2026-03-24 (see IDEAS.md appendix).

**Approach:** Fork-from-baseline. Every experiment is a standalone copy of `train_gpt.py` with targeted modifications. Each variant's `train_gpt.py` is a self-contained script that includes all necessary code changes — quantization plumbing, architecture changes, and export logic — so it can run independently without modifying the baseline.

**Prerequisite on H200:** FineWeb dataset downloaded (`python3 data/cached_challenge_fineweb.py --variant sp1024`).

---

## Interpreting 1xH200 Results

The 1xH200 setup is a **coarse filter**, not a precise ranking tool.

| Setting | 8xH100 (leaderboard) | 1xH200 (our experiments) |
|---------|----------------------|--------------------------|
| `nproc_per_node` | 8 | 1 |
| `grad_accum_steps` (derived: `8 // world_size`) | 1 | 8 |
| Effective batch (tokens/step) | 786,432 | 786,432 (same) |
| Wall time per step | ~84.5ms | ~600-700ms (est.) |
| Steps in 600s | ~7,100 | ~900-1,000 |
| FlashAttention | FA3 (Hopper) | FA3 (Hopper) — H200 is Hopper |

**What the 1xH200 run CAN tell us:**
- Clear winners and obvious losers (deltas > 0.01 BPB)
- Whether an architecture change breaks training stability
- Relative artifact size under different quantization schemes

**What it CANNOT reliably tell us:**
- Close-call rankings (deltas < 0.005 BPB) — these need 8xH100 validation
- Late-training dynamics: warmdown, late QAT, and EMA averaging all happen in the final ~50% of training steps. With ~900 steps instead of ~7100, these phases are compressed and may behave differently.
- Depth-sensitive ideas (12th layer, recurrence) may look worse than they truly are because they lose proportionally more steps to the per-step overhead.

**Rule of thumb:** Trust the direction of large deltas. Re-validate anything within 0.005 BPB on 8xH100 before making submission decisions.

---

## Directory Structure

```
experiments/
  v00_baseline/train_gpt.py           # Unmodified bug-fixed baseline (control)
  v01_xsa_all/train_gpt.py            # XSA all 11 layers
  v02_rebalance_8kv/train_gpt.py      # 8Q/8KV + 2.5x MLP
  v03_no_resid_mix_dec/train_gpt.py   # Remove resid_mix from decoder
  v04_trigram_conv/train_gpt.py       # Depthwise causal conv replaces SmearGate
  v05_no_value_embed/train_gpt.py     # Remove ValueEmbedding
  v06_12L_int5mlp/train_gpt.py        # 12 layers + int5 MLP quant [REQUIRES NEW QUANT CODE]
  v07_ushaped_quant/train_gpt.py      # U-shaped layerwise precision [REQUIRES NEW QUANT CODE]
  quant_frontier.py                    # Phase 1: sweep quant configs on baseline ckpt
  eval_stride32.py                     # Phase 2: re-eval baseline with stride=32
  eval_recurrence.py                   # Phase 2: two-pass eval on baseline ckpt
  run_all.sh                           # Master runner
  check_progress.sh                    # Dashboard for remote monitoring
  results/                             # Logs + metrics per experiment

checkpoints/
  .gitattributes                       # git-lfs tracking for large files
```

---

## Execution Order

### Queue A: Runs Executable Tonight (no new plumbing needed)

These experiments only change architecture, hyperparameters, or eval logic. The existing int6 quantization and export pipeline works unchanged.

#### Step 0: Train Baseline + Export Checkpoint (~12 min)

Train `v00_baseline/train_gpt.py` on 1xH200. The script produces:
- `final_model.pt` — full fp32 state dict
- `final_model.int6.ptz` — int6+zstd compressed artifact

The runner copies these to `checkpoints/` for reuse by Phase 1/2 tests:
- `checkpoints/baseline_raw.pt` ← copy of `final_model.pt`
- `checkpoints/baseline_compressed.ptz` ← copy of `final_model.int6.ptz`

Push both to GitHub via git-lfs for persistent comparison.

#### Step 1: Zero-Cost Eval Tests (~10 min) — IDEAS.md Phase 2

| Script | What it tests | Input | Expected time |
|--------|--------------|-------|---------------|
| `eval_stride32.py` | Sliding window stride=32 vs stride=64 | `baseline_compressed.ptz` | ~5 min |
| `eval_recurrence.py` | Two-pass forward at eval time (Section 3A) | `baseline_compressed.ptz` | ~5 min |

Both scripts load the compressed artifact, dequantize, and re-evaluate. No training.

#### Step 2: Architecture Ablations (~60 min) — IDEAS.md Phase 3

Run sequentially. Each is a full training run using `torchrun --standalone --nproc_per_node=1`.

Note: `grad_accum_steps` is derived internally by the script as `8 // world_size`. With `world_size=1`, this automatically becomes 8. No env var override needed.

All training runs use seed=1337, 600s wallclock, same data path.

Flag legend:
- `Ready` = runnable on the current train/export pipeline with straightforward model edits.
- `Infra` = blocked on new quantization/export/QAT plumbing.

| Order | Variant | IDEAS.md | Key Code Change | Risk | Flag |
|-------|---------|----------|----------------|------|------|
| 1 | v01_xsa_all | Section 9 | Set `XSA_LAST_N=11` env var (or hardcode in script) | Low | Ready |
| 2 | v02_rebalance_8kv | Section 7 | `NUM_KV_HEADS=8`, `MLP_MULT=2.5` | Medium | Ready |
| 3 | v03_no_resid_mix_dec | Section 4 | Delete `resid_mix` param from decoder Block, simplify forward | Low | Ready |
| 4 | v04_trigram_conv | Section 5 | New `TrigramConv` class replaces `SmearGate`, identity-init | Medium | Ready |
| 5 | v05_no_value_embed | Section 6 | Remove `ValueEmbedding` class and all VE injection in attention | Low | Ready |
| 6 | v08_engram_lite | Section 11A | Add context-aware gating to BigramHashEmbedding (W_K, W_V projections + RMSNorm gating) | Medium | Ready |
| 7 | v09_dca_scalar | Section 12A | DCA GRN-v1 scalar weights per prior layer, replace resid_mix, 55 new params | Medium | Ready |
| 8 | v10_dca_full | Section 12B | Full DCA GRN-v3 on block input, replace resid_mix + skip_weights, ~118K params | High | Ready |

These all export using the existing int6 quantization pipeline — no changes to `mixed_quantize_int6()`, `_classify_param()`, or `CastedLinear` QAT logic.

### Queue B: Runs Blocked on New Quantization/Export Code

These experiments require changes to the quantization and export pipeline before they can run. Each variant's `train_gpt.py` must include the new plumbing inline.

#### Required New Code (built into v06 and v07 scripts):

1. **Per-module `_clip_range` assignment** — During `GPT.__init__()`, set `module._clip_range = 15` (int5) or `module._clip_range = 7` (int4) on targeted `CastedLinear` modules. Must be compile-safe (class attribute, not instance attribute toggled at runtime).

2. **Mixed-precision `mixed_quantize()` dispatcher** — Extend `mixed_quantize_int6()` to route different components to different clip ranges based on param name. Currently it only distinguishes `{"mlp", "attn"}` vs other. Needs to support:
   - v06: `mlp → clip_range=15 (int5)`, `attn → clip_range=31 (int6)`
   - v07: layerwise `_get_clip_range(name, layer_idx, cat)` function per Section 1E

3. **QAT clamp alignment** — `CastedLinear.forward()` QAT branch must use `self._clip_range` instead of hardcoded 32/31. The hook exists but per-module assignment hasn't been wired. Clamp must be symmetric: `[-clip_range, clip_range]` for both QAT and export.

#### Step 3: Quantization Frontier Sweep (~5 min) — IDEAS.md Phase 1

Run `quant_frontier.py` on `checkpoints/baseline_raw.pt`. This script contains its own quantization functions (not importing from train_gpt.py) so it can test configurations that the training pipeline doesn't yet support.

Sweeps:

| Config | Clip range | What it tests |
|--------|-----------|--------------|
| Uniform int6 (current) | 31 | Control |
| Int5 MLP fc only | fc:15, proj:31 | Section 1A |
| Int5 all MLP (fc + proj) | mlp:15, attn:31 | Section 1B |
| Attn int8 + MLP int5 | attn:127, mlp:15 | Section 1D |
| Attn int8 + MLP int4 | attn:127, mlp fc:7 | Section 1D aggressive |
| U-shaped layerwise | varies by layer+component | Section 1E |

Output per config: compressed artifact bytes, post-quant BPB (non-sliding), post-quant BPB (sliding stride=64).

**This sweep directly informs whether v06 and v07 are worth building.** If int5 MLP kills BPB on the existing checkpoint, skip v06_12L_int5mlp.

#### Step 4: Training Experiments with New Quant Code (~24 min)

Only run after quant_frontier.py confirms the precision schemes don't destroy BPB.

| Order | Variant | IDEAS.md | Key Code Change | Risk | Flag |
|-------|---------|----------|----------------|------|------|
| 1 | v06_12L_int5mlp | Section 8 | `NUM_LAYERS=12`, int5 MLP export, per-module `_clip_range`, symmetric U-Net (6+6), VE layers → 10,11, XSA layers → 8-11 | High | Infra |
| 2 | v07_ushaped_quant | Section 1E | Layerwise `_get_clip_range()` in export, per-layer QAT clip ranges, int5 mid attn, int4 mid MLP fc, int8 late attn | High | Infra |

Both scripts include the new quantization plumbing inline. They are self-contained — they do not depend on changes to the root `train_gpt.py`.

---

## Progress Monitoring (`check_progress.sh`)

When run from any machine (e.g., local Mac via SSH), reports:

1. **Current status:** Which experiment is running, step count, elapsed time, latest train_loss
2. **ETA:** Estimated completion for current experiment + all remaining
3. **Completed table:** All finished experiments with final BPB, artifact size, and delta vs baseline

Reads from `experiments/results/` where each experiment writes:
- `{name}.log` — full stdout/stderr
- `{name}.json` — structured metrics (val_bpb, artifact_bytes, steps, elapsed_s)
- `progress.json` — updated by `run_all.sh` with current experiment name, index, and start time

---

## Checkpoint Management

- `checkpoints/baseline_raw.pt` — copied from `v00_baseline/final_model.pt` after training
- `checkpoints/baseline_compressed.ptz` — copied from `v00_baseline/final_model.int6.ptz` after training
- Both pushed to `github.com/ryanhartman4/parameter-golf` via git-lfs
- Winning experiments also get their checkpoints exported here

---

## Results Summary

`run_all.sh` prints a final comparison table after all experiments:

```
=== FINAL RESULTS ===
Experiment              | val_bpb  | Δ baseline | artifact_mb | notes
v00_baseline            | X.XXXX   | --         | XX.XX MB    |
eval_stride32           | X.XXXX   | ±X.XXXX   | --          | eval only
eval_recurrence         | X.XXXX   | ±X.XXXX   | --          | eval only
v01_xsa_all             | X.XXXX   | ±X.XXXX   | XX.XX MB    |
v02_rebalance_8kv       | X.XXXX   | ±X.XXXX   | XX.XX MB    |
v03_no_resid_mix_dec    | X.XXXX   | ±X.XXXX   | XX.XX MB    |
v04_trigram_conv        | X.XXXX   | ±X.XXXX   | XX.XX MB    |
v05_no_value_embed      | X.XXXX   | ±X.XXXX   | XX.XX MB    |
--- Queue B (if quant frontier passes) ---
quant_frontier          | (table)  | (table)    | (table)     | sweep results
v06_12L_int5mlp         | X.XXXX   | ±X.XXXX   | XX.XX MB    | needs new quant code
v07_ushaped_quant       | X.XXXX   | ±X.XXXX   | XX.XX MB    | needs new quant code
```

---

## Known Confounds (not bugs — inherent to the experiments)

These were flagged during code review and are accepted design tradeoffs:

| Experiment | Confound | Why it's accepted |
|-----------|----------|-------------------|
| **v04_trigram_conv** | Conv kernel uses AdamW (scalar_params) instead of Muon. SmearGate's gate also used AdamW, but this is a 3D kernel vs a 1D gate — different optimizer dynamics. | Muon crashes on 3D tensors (`zeropower_via_newtonschulz5` uses `.T` and `@`). AdamW is the only safe choice. The confound is small since SmearGate was also on AdamW. |
| **v09_dca_scalar** | Maintains a full hidden-state buffer (~1 GiB in bf16 per microbatch, 11 layers x batch x seq x 512). Adds memory pressure and may reduce throughput. | Inherent to DCA — you must store prior layer outputs to compute weighted combinations. H200 has 141GB, so 1GB is <1%. Throughput impact is the real cost but unavoidable for this architecture. |
| **v10_dca_full** | Same buffer issue as v09, plus GRN composition (einsum over prior layers) at every block. Larger systems-level overhead. | Same rationale as v09. The GRN einsum is the core DCA operation — can't test DCA without it. If v09 (scalar-only, cheaper) already shows a win, v10 tells us whether the full GRN is worth the extra overhead. |

When interpreting results: if v09/v10 show a regression, it may be the throughput cost (fewer steps in 600s) rather than the architecture itself being worse. Check step counts in the results JSON.

---

## After Overnight Run — Next Steps

1. Analyze results table. Identify winners (negative Δ vs baseline).
2. For close calls (< 0.005 BPB delta), flag as "needs 8xH100 validation."
3. **Phase 5 (IDEAS.md):** Stack the clear winners into a combined variant.
4. Validate the best combined variant on rented 8xH100 with 3 seeds.
5. If it beats 1.1228 by >=0.005 with p<0.01, prepare a leaderboard submission PR.

