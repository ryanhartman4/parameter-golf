# Parameter Golf — Ideas for Beating 1.1228 BPB

Current SOTA: **1.1228 BPB** (signalrush, 11L EMA + GPTQ-lite + warmdown3500)
Architecture: 11 layers, 512 dim, 8Q/4KV heads, 3x MLP, 1024 vocab, ~27M params → 15.5MB

### Two Independent Bottlenecks

Every idea in this doc hits one or both constraints:

| Constraint | Budget | What helps | What hurts |
|---|---|---|---|
| **Artifact size** | 16 MB (code + compressed model) | Lower precision, better compression, fewer params | More layers, wider model, higher precision |
| **Wallclock** | 600s training on 8xH100 (~7100 steps at 84.5ms/step) | Faster steps, fewer layers per step | More layers (+9% per layer), wider model (~1.56x for 512→640), heavier ops |

Ideas that "free MB" do not automatically save wallclock. Proposals that add layers or width must account for step-time increase and the resulting loss of training steps.

*All line references below are for the root `train_gpt.py` (SOTA copy + bug fixes applied 2026-03-24).*

*Prerequisites (4 bugs + 2 efficiency fixes) are complete — see [Appendix: Completed Fixes](#appendix-completed-fixes) at the bottom.*

---

## 1. Mixed Int5/Int6 Quantization (Three Tiers)

### Strategy A: Conservative — Int5 MLP fc only

Quantize only the MLP expansion matrices (512→1536) to int5. Keep everything else at int6.

| What changes | Layers affected | Size saved |
|---|---|---|
| `mlp.fc` weights → int5 | All 11 layers | **~1.0 MB** |
| Everything else stays int6 | — | — |

**Buys:** ~1.3M extra int6 params — enough for wider bigram (10K buckets) or slightly wider MLPs.

Risks:
- The savings are meaningful but still modest; this alone does not fund a full 12th layer.
- The evidence is indirect: submission #5 validated int5 MLP in a nearby architecture, not this exact 11-layer checkpoint.

Mitigations:
- This is the cleanest checkpoint-only re-quantization test in the whole doc.
- If it works, use it as the baseline mixed-precision recipe before touching `mlp.proj`, attention, or int4.

### Strategy B: Moderate — Int5 all MLP, int6 attention (RECOMMENDED)

Copy submission #5's approach on the SOTA's 11-layer architecture.

| What changes | Layers affected | Size saved |
|---|---|---|
| `mlp.fc` + `mlp.proj` → int5 | All 11 layers | **~2.1 MB** |
| Attention stays int6 | — | — |
| Last-layer K proj → FP16 | Layer 10 | negligible |

**Buys:** ~2.7M extra int6 params — enough for a **12th transformer layer** (2.4M params).

Risks:
- `mlp.proj` is more sensitive than `mlp.fc` because it writes directly back into the residual stream.
- QAT has to match int5 export semantics or the late-training signal becomes misleading.
- Per-module clip ranges are more complicated under compiled training than checkpoint-only export.

Mitigations:
- Validate this as a zero-cost re-quantization pass on an existing checkpoint before any new training.
- Keep the int5 convention symmetric `[-15, 15]` so QAT and export agree.
- Treat this as the default mixed-precision path only if the checkpoint sweep confirms both bytes and BPB.

**Code change locations:**
- `_classify_param()` at **line 887** — already categorizes "mlp" vs "attn". No change needed here.
- `mixed_quantize_int6()` at **line 920** — the dispatch loop. Add clip_range routing where `cat in int6_cats`:
```python
if cat == "mlp":
    q, s = quantize_int6_per_row(t, clip_range=15)  # int5
else:
    q, s = quantize_int6_per_row(t)  # int6 (default clip_range=31)
```
- `quantize_int6_per_row()` at **line 895** — already accepts `clip_range` as a parameter. No change needed.
- `CastedLinear.forward()` QAT branch at **lines 423-427** — Bug 1 is fixed (two-phase compile + symmetric clamp). For int5 MLP modules, the next step is to set `self._clip_range = 15` during `GPT.__init__()`. **Note:** current code exposes the `_clip_range` hook, but per-module assignment is still future work. QAT clamp must be symmetric `[-15, 15]` (not `[-16, 15]`) to match the export quantizer's `[-clip_range, clip_range]` convention.

### Strategy C: Aggressive — Mixed int4/int5/int6 by layer

| What changes | Layers | Size saved |
|---|---|---|
| MLP fc → int4 | Layers 0-4 (encoder) | ~1.2 MB |
| MLP fc → int5, proj → int5 | Layers 5-10 (decoder) | ~1.3 MB |
| Attention → int6 | All layers | — |
| **Total** | | **~2.5-3.0 MB** |

**Buys:** ~3.5-4M extra params — 12th layer + wider MLP or wider attention.

Risks:
- Int4 is still unproven here. With the current symmetric clamp it only has 15 levels, so mistakes get large quickly.
- This likely wants stronger quantization machinery than the current percentile search.
- A layerwise recipe this aggressive is hard to debug because many changes move at once.

Mitigations:
- Start with `mlp.fc` only; keep attention at int6 until the int4 tolerance is real.
- Move to this only after Strategy B is working and the quantization frontier harness shows clear headroom.
- If int4 is pursued, pair it with earlier QAT and a stronger GPTQ variant.

Why it might work:
- ReLU-squared makes MLP activations sparse enough that some weight error may wash out.
- MLP weights are structurally simpler than attention weights.
- The fc expansion matrix is the most redundant matrix in the block.

### Strategy D: Raise Attention, Drop MLP (Zero-Sum Swap) NEW

**Core idea:** Instead of uniform precision, give attention more bits while taking them from MLP. If attention is the more sensitive subsystem, this could improve quality without growing the artifact.

**Current (uniform int6) — estimates are post-zstd compressed:**
| Component | Precision | Raw (int8 bytes) | Zstd ratio | Compressed est. |
|---|---|---|---|---|
| Attention (8.65M params) | Int6 [-31,31] | 8.65 MB | ~1.51× | **~5.7 MB** |
| MLP (17.3M params) | Int6 [-31,31] | 17.3 MB | ~1.51× | **~11.5 MB** |

**Proposed (attention int8 + MLP int5):**
| Component | Precision | Raw (int8 bytes) | Zstd ratio | Compressed est. | Delta |
|---|---|---|---|---|---|
| Attention (8.65M params) | **Int8** [-127,127] | 8.65 MB | ~1.25× (est.) | **~6.9 MB** | +1.2 MB |
| MLP (17.3M params) | **Int5** [-15,15] | 17.3 MB | ~1.88× | **~9.2 MB** | -2.3 MB |

**Buys:** In the optimistic estimate, attention int8 + MLP int5 frees ~1.1 MB while improving attention precision.

**Aggressive version (attention int8 + MLP int4):**
| Component | Precision | Raw | Zstd ratio | Compressed est. | Delta |
|---|---|---|---|---|---|
| Attention (8.65M params) | **Int8** | 8.65 MB | ~1.25× (est.) | **~6.9 MB** | +1.2 MB |
| MLP (17.3M params) | **Int4** [-7,7] | 17.3 MB | ~2.4× (est.) | **~7.2 MB** | -4.3 MB |

**Buys:** In the aggressive estimate, attention int8 + MLP int4 could free ~3.1 MB.

Risks:
- The compression ratios for int8 and int4 are estimates, not measurements.
- Attention may be more precision-sensitive, but the gain from int8 still has to beat the value of the params it displaces.
- Mixed per-component precision increases QAT complexity.

Mitigations:
- Treat this as a checkpoint-sweep idea first, not a training plan.
- Measure actual zstd-22 bytes for attention int8 before spending any design effort on it.
- Start with the milder `attention int8 + MLP int5` split before even considering int4.

### Strategy E: U-Shaped Precision Curve (Layerwise + Component Grid) NEW

**Core idea:** Make precision follow a U-shape by layer: higher at the edges, lower in the middle. Early layers set up the representation, late layers drive logits, and middle layers may be the most tolerant of quantization noise.

```
Precision
  high ██                                    ██
  med  ████                              ██████
  low  ██████ ██████████████████████ ██████████
       L0  L1  L2  L3  L4  L5  L6  L7  L8  L9  L10
       ── early ──  ──── middle ────  ── late ───
```

**Full precision grid:**

| Layer | Attention | MLP fc | MLP proj | Rationale |
|---|---|---|---|---|
| 0-2 (early) | Int6 | Int5 | Int5 | Building features that propagate; moderate tolerance |
| 3-7 (middle) | **Int5** | **Int4** | **Int5** | Most tolerant zone — 4-7 corrective layers ahead |
| 8-9 (late) | **Int8** | Int5 | **Int6** | Late attention sensitive; proj feeds residual stream |
| 10 (final) | **Int8** | Int5 | **Int6** | Last layer directly drives logits |
| Last K proj | **FP16** | — | — | Proven sensitive by submissions #5 and #6 |

**Buys:** This opens a new axis that nobody seems to be using yet: layerwise attention precision instead of treating all attention weights as equally sacred.

**Size estimate vs uniform int6:**

| Zone | Attention delta | MLP delta |
|---|---|---|
| Early (3 layers) | ±0 (stays int6) | saves ~0.3 MB (int5 MLP) |
| Middle (5 layers) | saves ~0.4 MB (int5 attn) | saves ~1.2 MB (int4 fc + int5 proj) |
| Late (3 layers) | costs ~0.5 MB (int8 attn) | saves ~0.15 MB (int5 fc, int6 proj mixed) |
| **Net** | ~-0.1 MB | ~-1.35 MB |
| **Total savings** | | **~1.5 MB freed** |

That is ~1.9M extra int6 params — not quite a full 12th layer but meaningful.

Risks:
- This stacks several assumptions at once: middle-layer attention tolerance, late-layer attention sensitivity, and int4 tolerance in middle MLP fc.
- The strongest evidence for the middle-layer story comes indirectly from XSA, not from direct quantization tests on this architecture.
- Layer-specific QAT and export logic become more complex very quickly.

Mitigations:
- Validate this through a quantization frontier harness before training anything.
- Start with a simpler variant: all MLP int5, late attention int8, no int4.
- Only add middle-layer attention int5 after checkpoint sweeps show it is not obviously harmful.

**Implementation:**
- In `mixed_quantize_int6()` (**line 920**): extract layer index from param name (`blocks.N.` prefix). Build a `clip_range_map` dict keyed by `(layer_idx, component)`:
```python
def _get_clip_range(name, layer_idx, cat):
    if layer_idx is None: return 31  # non-layer params
    if 3 <= layer_idx <= 7:  # middle zone
        if cat == "attn": return 15   # int5 attention
        if cat == "mlp" and ".fc." in name: return 7  # int4 MLP fc
        if cat == "mlp": return 15    # int5 MLP proj
    if layer_idx >= 8:  # late zone
        if cat == "attn": return 127  # int8 attention
        if cat == "mlp" and ".proj." in name: return 31  # int6 MLP proj
        if cat == "mlp": return 15    # int5 MLP fc
    return 31 if cat == "attn" else 15  # early: int6 attn, int5 MLP
```
- In `CastedLinear.forward()` QAT branch (**line 425**): use `self._clip_range` instead of hardcoded 32/31. The hook exists today; setting `_clip_range` per module during `GPT.__init__()` is still future work and needs to be done in a compile-safe way.
- **~30 lines of changes** across two functions, but verify each layer's precision matches the grid.

---

## 2. Precision vs Parameters Tradeoff (Key Insight)

**Parameters win over precision.** From scaling laws analysis:
- Cutting 27M → 20M params (to fit int8): costs ~0.023 BPB
- Saving from int6 → int8 precision: gains only ~0.006 BPB
- **Net: int8 would be 0.017 BPB worse than int6**

Rule of thumb: each bit of entropy reduction across MLP params saves ~2.2 MB, funding ~3M additional int6 params per bit dropped.

Current SOTA quantization error (non-sliding, apples-to-apples):
- Pre-quant (post-EMA): 1.1385 BPB
- Post int6 roundtrip: 1.1466 BPB
- **Quantization penalty: +0.0081 BPB** (reduced from 0.016 by GPTQ-lite + QAT)

---

## 3. Depth Recurrence

### Critical Discovery: Eval Has 500+ Seconds Free

| Phase | Time Used | Budget | Headroom |
|---|---|---|---|
| Training | 600s | 600s | **0s** (maxed out) |
| Evaluation | ~82s | 600s | **~518s free** |

Training runs at ~84.5ms/step for ~7100 steps — fully maxed. But evaluation (sliding window stride=64) only takes ~73s + ~9s overhead = ~82s of a 600s budget. **7× headroom for eval-time compute.**

### Approach A: Eval-Only Recurrence (Lowest Risk)

Train normally (11 layers, 7100 steps). At eval time, run transformer blocks **twice** for 22 effective depth.

**Buys:** Zero training cost. Eval time roughly doubles from ~73s to ~146s, which still fits comfortably under the 600s eval budget.

Risks:
- The model was not trained to see a second pass through the same stack.
- Existing repo evidence already says **train-time** recurrence can be very bad under a fixed wallclock budget, so recurrence should not be treated as a default training strategy.
- Eval recurrence may still produce no gain or even hurt if the second pass amplifies errors instead of correcting them.

Mitigations:
- Keep this strictly eval-only at first.
- Test it on an existing checkpoint before changing any training setup.
- Measure both BPB and eval time at stride 64 and stride 32 before promoting it.

**Implementation:** Modify `GPT.forward()` (**line 743**) or create a `forward_logits_recurrent()` variant. The forward pass has a clear structure: embedding (lines 744-749), encoder loop (lines 752-755), decoder loop (lines 756-761), final norm + logits (lines 762-771). To add a second pass:
- After the decoder loop completes, re-enter the encoder loop at line 752 with the current `x` (skips list starts empty again, encoder re-pushes)
- Then re-enter the decoder loop at line 756
- `x0` persists from the original embedding — resid_mix naturally re-injects it
- XSA (`_xsa_efficient` at **line 511**) operates on the current `v` tensor per call — no state to reset

For eval-only recurrence, only modify the compiled `forward_logits` (**line 790**), not the training forward. Compile with `torch.compile` once and reuse.

**Verdict:** Cheap to test, but not something to build the plan around until the checkpoint result is clearly positive.

### Approach B: Weight-Shared Layers

Instead of 11 unique layers, use **6 unique + 5 shared:**

```
Layer 0: unique (W0)     Layer 6: reuses W0
Layer 1: unique (W1)     Layer 7: reuses W1
Layer 2: unique (W2)     Layer 8: reuses W2
Layer 3: unique (W3)     Layer 9: reuses W3
Layer 4: unique (W4)     Layer 10: reuses W4
Layer 5: unique (W5)
```

**Buys:** Roughly halves the stored transformer weights while still executing the same number of block applications. The main value is artifact budget, not training speed.

**Why this works:** Each shared layer processes different inputs (after 5-6 layers of transformation). Same weights do different work at different depths. This is implicit recurrence — the model learns features useful at multiple depths.

Risks:
- Weight sharing saves bytes and optimizer state, but it does **not** save FLOPs. The model still runs every block application every step.
- U-Net makes encoder and decoder roles asymmetric, so sharing across the midpoint may be unnatural.
- XSA may create another asymmetry if a shared block is used once without XSA and later with XSA.

Mitigations:
- Keep `model_dim=512` for the first prototype; do not bundle width expansion into the first test.
- Share within same-role regions first, such as encoder-to-encoder or decoder-to-decoder, before trying encoder/decoder mirrors.
- Keep `resid_mix`, `attn_scale`, `mlp_scale`, and skip control parameters per-position while sharing only the heavy linear weights.

**Implementation:** In `GPT.__init__()` (**lines 634-710**), the block list is built at **line 676** as:
```python
self.blocks = nn.ModuleList()
for i in range(num_unique_layers):
    self.blocks.append(Block(..., layer_idx=i))
# Shared layers reference existing blocks (no new params)
for i in range(num_shared_layers):
    self.blocks.append(self.blocks[i])  # same Module object
```
The `nn.ModuleList` will contain references to shared modules — PyTorch only counts params once. The forward loop in `GPT.forward()` (**lines 752-761**) doesn't change — it just iterates over `self.blocks` as before, but some entries point to the same weights.

### Approach C: Train-Time Recurrence with Curriculum (Highest Risk/Reward)

- Steps 0-4500: Train with 11 layers (84.5ms/step, fast convergence) = ~380s
- Steps 4500+: Switch to 2 passes through all 11 layers (~160ms/step) = ~220s remaining
- Add learned pass embedding `[2, 512]` so model distinguishes passes (Universal Transformer style)

**The math:** ~4500 fast steps × 84.5ms = 380s. Remaining: 220s / 160ms = **~1375 slow steps** (not 1600). Total: ~5875 steps. Model gets 4500 steps of fast learning, then ~1375 steps to learn recurrence.

Risks:
- This gives up a large number of training steps under the fixed wallclock.
- Curriculum switches are easy to destabilize.
- There is already a negative recurrence result in the repo, even if it is not this exact variant.

Mitigations:
- Keep this as a separate high-risk branch, not part of the mainline plan.
- Start the second pass late and isolate it behind a learned pass embedding.
- Re-tune warmdown specifically for the slower late phase instead of reusing the current schedule.

### Combo Play: Weight Sharing + Eval Recurrence

**Idea:** If weight sharing works at all, it naturally pairs with eval-time recurrence because both try to get more effective depth out of fewer stored weights.

Risks:
- This compounds two already-unproven ideas.
- The interaction is hard to reason about because both change how depth is represented.

Mitigations:
- Only consider this after standalone wins from weight sharing and eval recurrence.
- Keep the first combo small: shared weights at 512 dim plus one extra eval pass.

---

## 4. Remove/Simplify Residual Mixing (resid_mix)

**Observation:** Ryan's experiments show x0 mixing at every layer doesn't help much.

**Idea:** Trim or simplify `resid_mix`, especially in decoder layers, and see if the rest of the architecture is already carrying enough early information forward.

Why it might be redundant:
1. resid_mix (x0 at every layer) ← candidate for removal
2. U-Net skip connections (encoder states to decoder)
3. BigramHash embedding (baked into x0)
4. Value Embedding (token identity at layers 9-10)

**Buys:** Very small parameter savings, but possibly cleaner block behavior and one less cross-layer mechanism to reason about.

Risks:
- The redundancy story is only partial. U-Net skips carry processed states, not raw `x0`.
- BigramHash is only injected once at input.
- Value embedding only exists on a couple of late layers.
- For only ~11K params, this is more of a simplification bet than a budget unlock.

Mitigations:
- Start with decoder-only removal instead of deleting `resid_mix` everywhere.
- Keep an encoder-side scalar gate variant if full removal is too abrupt.
- Treat this as an ablation question, not a default cleanup.

**Proposed changes:**
- **Remove resid_mix entirely from decoder layers (5-10)** — U-Net skips serve this purpose
- **Encoder layers (0-4):** either remove entirely, or replace with a single scalar lambda per layer (5 params, init=0)
- **Reinvest** the architectural simplification into cleaner gradient flow for deeper models

**Code locations:**
- `resid_mix` is a `nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]))` at **line 616** in `Block.__init__()`
- Used at **lines 625-626** in `Block.forward()`: `mix = self.resid_mix; x_in = mix[0]*x + mix[1]*x0`
- `x0` is saved at **line 749** in `GPT.forward()` and passed to every block call
- To remove: delete the `resid_mix` parameter from Block, change `Block.forward()` to just use `x` directly, remove `x0` argument from decoder block calls (lines 756-761). Keep `x0` for encoder blocks if using scalar lambda variant.

This also becomes more attractive if weight sharing is used, because shared blocks then have fewer role-dependent pathways to reconcile.

---

## 5. Depthwise Trigram Conv (Replace SmearGate)

**Idea:** Replace `SmearGate` (current: `SmearGate` at **lines 540-546**, used at **line 748** in `GPT.forward()`) with a tiny causal depthwise trigram conv.

Instead of SmearGate (look-back-1, 512 params) or trigram hash table (770K+ params), use:
```python
# Causal depthwise conv — pad left only, no future leakage
self.local_conv = nn.Conv1d(512, 512, kernel_size=3, padding=0, groups=512)

def forward(self, x):
    # x: [B, T, 512] → transpose → pad left by 2 → conv → transpose back
    x = x.transpose(1, 2)                    # [B, 512, T]
    x = F.pad(x, (2, 0))                     # left-pad only (causal)
    x = self.local_conv(x)                    # [B, 512, T]
    return x.transpose(1, 2)                  # [B, T, 512]
```
**Buys:**
- Captures trigram-like local patterns in continuous embedding space.
- Uses only 1,536 params if `bias=False`.
- Still stays causal with left-only padding.

Risks:
- `nn.Conv1d` defaults to `bias=True`, which makes it 2,048 params unless changed.
- Unlike `SmearGate`, this is not naturally identity-initialized or gated.
- It may be noisier early in training even if it is more expressive later.

Mitigations:
- Use `bias=False` if the goal is the minimal version.
- Identity-init the center tap and zero-init the side taps so the module starts close to a no-op.
- Compare it directly against `SmearGate` in a small ablation instead of assuming it is strictly better.

---

## 6. Simplify-and-Deepen Strategy

**Idea:** As mechanisms accumulate, some of the older helper paths may no longer be pulling their weight. Prune a few of them and reinvest the bytes or the conceptual simplicity elsewhere.

Candidates for removal/simplification:
| Mechanism | Params | Redundant Because |
|---|---|---|
| resid_mix (decoder layers) | 6K | U-Net skips do this better |
| SmearGate | 512 | Depthwise conv may be better (needs identity-init, see §5 caveats) |
| Value Embedding | ~164K | U-Net skips + bigram provide token identity |
| resid_mix (encoder layers) | 5K | Marginal value per Ryan's experiments |

**Total freed:** ~172K params plus some forward-pass simplification.

Risks:
- None of these items is very large on its own, so small regressions can overwhelm the savings.
- Bundling several removals together makes attribution hard.

Mitigations:
- Ablate one simplification at a time.
- Treat the savings as a bonus, not the main reason to do it.
- Reinvest only after one simplification is clearly neutral or positive.

---

## 7. Rebalance MLP:Attention Ratio (More/Wider KV Heads)

**Observation:** The current model is 64% MLP, 32% attention. This 2:1 ratio is an artifact of the 3× MLP expansion — inherited from winning submissions without being independently tuned.

**Current attention config:** 8 query heads, 4 KV heads, head_dim=64.

**The question:** Is the model bottlenecked on attention capacity? With only 4 KV heads, the model has 4 independent "views" of the context. More KV heads = richer relational modeling between tokens.

### Option 1: More KV capacity at fixed params

Spend some MLP budget on richer attention instead of trying to grow the whole model. The cleanest valid version at `model_dim=512` is:

| Change | Param delta per layer | Total (11 layers) |
|---|---|---|
| `c_k`: 512×256 → 512×512 | +131,072 | +1,441,792 |
| `c_v`: 512×256 → 512×512 | +131,072 | +1,441,792 |
| **Attention total** | **+262,144/layer** | **+2,883,584** |

Fund it by reducing MLP from `3.0×` → `2.5×` (hidden `1536 → 1280`):

| Change | Param delta per layer | Total (11 layers) |
|---|---|---|
| `fc`: 512×1536 → 512×1280 | -131,072 | -1,441,792 |
| `proj`: 1536×512 → 1280×512 | -131,072 | -1,441,792 |
| **MLP total** | **-262,144/layer** | **-2,883,584** |

**Net:** zero extra params, but the ratio shifts toward attention via `8Q/8KV` instead of `8Q/4KV`.

Risks:
- The original `4 → 6 KV` idea is not valid under `8Q`; GQA needs `num_q_heads % num_kv_heads == 0`.
- More KV capacity may still be the wrong trade if this model is mostly MLP-limited.

Mitigations:
- Use `8Q/8KV` as the concrete current-dim test instead of inventing a partial-GQA config that does not divide.
- Run it as a small-scale head-to-head against the current `8Q/4KV, 3.0× MLP` recipe.

### Option 2: Wider heads (head_dim 64 → 80)
Requires model_dim increase from 512 → 640. This affects **all** weight matrices, not just attention, so it is really a wider-model idea rather than an isolated head-size tweak.

Risks:
- Wallclock goes up quickly because the whole model widens.
- It is easy to confuse this with a pure attention experiment when it is actually a global scaling change.

Mitigations:
- Keep this separate from the fixed-dim KV-capacity test.
- Only pursue it if weight sharing or another artifact-saving move is already working.

### Which helps more?

The safe bet is still **keep the ratio and add a 12th layer**. The speculative bet is that more KV capacity gives the model richer context modeling at the same param count.

**Test plan:** Train two configs head-to-head on MLX (small scale):
1. Current: 8Q/4KV, 3.0× MLP
2. Rebalanced: 8Q/8KV, 2.5× MLP (same param count)

---

## 8. 12th Layer via Int5 MLP Precision Drop

**The straightforward play:** Keep the current architecture ratio, drop MLP to int5, use the freed 2.1MB for a 12th transformer layer.

**What changes:**
| Component | Before | After |
|---|---|---|
| MLP quantization | Int6 | Int5 |
| Compressed MLP size | ~13.0 MB | ~10.8 MB |
| Space freed | — | ~2.1 MB |
| Layer count | 11 | 12 |
| New layer params | — | ~2.4M (786K attn + 1.57M MLP + 2K control) |
| New layer quantization | — | Int5 MLP, int6 attn |

**Why 12th layer > wider MLP:**
- Adds both MLP AND attention capacity together (keeps ratio balanced)
- Deeper models consistently outperform wider ones at this scale (9→10→11 layers each improved BPB)
- The U-Net structure naturally accommodates 12 layers (6 encoder + 6 decoder, perfectly symmetric)
- Extra layer means extra XSA candidate (could apply to layers 8-11 instead of 7-10)

**U-Net at 12 layers:**
```
Encoder: layers 0-5 (6 layers, push 6 skips)
Decoder: layers 6-11 (6 layers, pop 6 skips)
Perfectly symmetric — fixes the current 5/6 asymmetry
```
**Hidden benefit:** With 11 layers, `num_skip_weights = min(5, 6) = 5` — layer 10 (the final decoder layer, which directly drives logits) currently runs WITHOUT a skip connection. At 12 layers: `min(6, 6) = 6` skips — every decoder layer gets one. This is a meaningful architectural fix, not just symmetry.

Risks:
- A 12th layer adds ~9% more compute per step, which likely costs roughly ~600 training steps inside the same 600s wallclock.
- The bet only works if the extra depth pays back more than the lost steps.
- Bundling it immediately with other changes makes attribution hard.

Mitigations:
- Pair it with the simplest proven artifact-saving move, namely int5 MLP.
- Run it as a mostly isolated full-training test before stacking multiple architectural edits.
- Keep the rest of the architecture close to the current SOTA when evaluating the first 12-layer attempt.

**Implementation:**
- Add a 12th `Block` in `GPT.__init__()` (around **line 676** where the `nn.ModuleList` of blocks is created)
- Update `num_layers` env var from 11 → 12
- U-Net split changes automatically: `num_encoder_layers = 12 // 2 = 6`, `num_decoder_layers = 6`
- `skip_weights` grows from `[5, 512]` to `[6, 512]` (one new parameter tensor)
- XSA config: update `XSA_LAST_N=4` to apply to layers 8-11 (or test XSA_LAST_N=5 for layers 7-11)
- LN scale factor auto-adjusts (it's `1/sqrt(layer_idx+1)`, no change needed)
- `ve_layers` default "9,10" should probably become "10,11" (last two layers)
- Quantization: the new layer's MLP gets int5, attention gets int6 (Strategy B from §1)

This does **not** require weight sharing, which is why it remains the straightest next full-training shot.

---

## 9. XSA All Layers — Promising but Needs Revalidation

**Idea:** Push XSA from the last 4 layers to all layers if the same regularizing effect carries over to this architecture.

Results from Ryan's nanochat d12 ablation (124M params, full MHA, different architecture):

| Config | Final BPB | Δ vs baseline | Throughput cost |
|---|---|---|---|
| Baseline (no XSA) | 0.8539 | — | — |
| **XSA all layers** | **0.8516** | **-0.0024** | -3.3% tok/sec |
| XSA last 4 | 0.8532 | -0.0007 | -1.9% tok/sec |

**Buys:** The nanochat signal is real enough to make this worth a direct test, and it may pair naturally with more aggressive quantization if it reduces redundant self-value components.

Risks:
- The evidence is from a different model family: full MHA, larger scale, and no U-Net skips, `resid_mix`, or bigram path.
- Under GQA, XSA is more coupled because two query heads share one value direction.
- Early-layer XSA may be too aggressive before heads have specialized.

Mitigations:
- Revalidate this directly on the parameter-golf architecture before treating it as real.
- Compare `XSA-last-4` vs `XSA-all` in a small-scale MLX ablation instead of guessing from nanochat.
- Only connect it to Strategy E or weight sharing if it first wins on its own.

---

## 10. Parked: Tokenizer Expansion (Too Risky for Now)

**Idea:** Expand the vocab from 1024 to 2048 or higher. Tokenizer training is offline, so in principle this can reduce sequence length and improve both training efficiency and eval context coverage.

Risks:
- Tokenizer changes will get much heavier scrutiny from the challenge organizers.
- BPB calculation becomes easier to get subtly wrong.
- A verification failure here would waste more time than most model-side dead ends.

Mitigations:
- Keep it parked until safer architecture and quantization ideas are exhausted.
- If revisited, build the verification story first, not after the training run.
- Treat the stock 1024 tokenizer as the default until there is a very strong reason to move.

---

## 11. Engram: Conditional Memory for Early-Layer Depth (arXiv:2601.07372)

**Paper:** Cheng et al. (DeepSeek, 2025). "Conditional Memory via Scalable Lookup." [[summary]](knowledge/summary_engram_conditional_memory.md)

**Core insight:** Transformers waste early layers reconstructing static N-gram patterns through compute. Engram replaces this with O(1) hash-based lookups into learned embedding tables, injected at layer 2 via context-aware gating. CKA analysis shows Engram layer 5 ≈ baseline layer 12 — effectively free depth. The paper validates a U-shaped scaling law: ~20-25% of spare parameter budget should go to memory, not more experts/layers.

**Why it fits Parameter Golf:** Our `BigramHashEmbedding` is already a primitive Engram (XOR hash into embedding table, learned scale). Upgrading it with multi-head hashing and context-aware gating could give us effective depth gain without adding transformer blocks (~9% wallclock per layer).

**Blocker:** Engram's power comes from *learned* embeddings that must be stored in the artifact. The paper uses billions of params; we have ~0.5MB headroom (or ~2.6MB after int5 MLP savings from §1B).

### Strategy 11A: Engram-Lite — Gated BigramHash (Zero-Cost Test)

Keep the existing XOR hash table (zero storage cost), but add learned context-aware gating.

| What changes | Storage cost | Wallclock cost |
|---|---|---|
| Add W_K, W_V projections (512 × bigram_dim) to BigramHashEmbedding | ~48KB at int6 | ~negligible (two small matmuls per step) |
| Gate = sigmoid(RMSNorm(h_t)^T * RMSNorm(W_K * e_t) / sqrt(d)) | — | — |
| Replace current additive injection with gated residual | — | — |

**Buys:** The model can suppress irrelevant hash lookups (collision noise, polysemy). The paper shows gating is one of the top-3 most impactful components.

Risks:
- The XOR hash table is deterministic and not learned — gating alone may not provide enough signal to compensate for hash collision noise at our small table size.
- The existing BigramHash already learns a scale factor per position; gating may be redundant.

Mitigations:
- This is a zero-cost test on an existing checkpoint: add gating projections, initialize to pass-through, measure BPB.
- If it helps, it's free to keep. If not, we lose nothing.

**Code change locations:**
- `BigramHashEmbedding` class (~line 550) — add `self.gate_k` and `self.gate_v` as `nn.Linear(bigram_dim, bigram_dim)`, add `self.gate_norm_h` and `self.gate_norm_k` as `RMSNorm()`.
- `BigramHashEmbedding.forward()` — compute gate alpha from hidden state and key projection, apply to value before residual add.
- `mixed_quantize_int6()` — ensure gating projections are included in quantization dispatch.

### Strategy 11B: Micro-Engram — Learned Embeddings at Int4 (Requires Int5 MLP First)

Replace BigramHashEmbedding with a proper Engram module using small learned embedding tables.

| What changes | Storage cost | Wallclock cost |
|---|---|---|
| 2-gram table: 3K slots × dim 48, 2 hash heads, int4 quantized | 3K × 48 × 0.5 = ~72KB per head = **~144KB** | Small — 2 table lookups + concat |
| W_K + W_V projections (512 × 96) | ~48KB at int6 | Negligible |
| Context-aware gating + depthwise conv (kernel 4) | ~2KB | Negligible |
| **Total** | **~194KB** | **<1ms/step est.** |

**Buys:** Learned embeddings capture actual N-gram statistics, not just hash artifacts. Multi-head hashing reduces collisions (two independent hash functions into prime-sized tables).

Risks:
- Int4 quantization of embedding tables is unproven — embeddings may need higher precision than weights.
- 3K slots for 1024^2 ≈ 1M possible 2-grams means ~333x collision rate. Multi-head hashing helps but doesn't eliminate this. The very high collision rate may limit how much signal the table can store.
- Requires freeing ~200KB from somewhere (int5 MLP from §1B provides 2.1MB, more than enough).

Mitigations:
- Start with int6 embeddings during training, quantize to int4 at export. If int4 hurts, fall back to int5 (~225KB total for tables).
- Use prime-sized tables (e.g., 3001, 3011) to reduce systematic collision patterns.
- Test on MLX first with small training runs before committing to a full 8xH100 run.
- Note: larger tables (e.g., 30K slots) are feasible if funded by int5 MLP savings (~2.1MB freed), but then this merges into Strategy 11C.

**Code change locations:**
- New class `EngramModule` replacing `BigramHashEmbedding` — multi-head hash lookup, W_K/W_V gating, optional depthwise conv.
- `GPT.__init__()` — instantiate EngramModule at layer 2 (or layer 0 pre-attention).
- `GPT.forward()` — inject Engram residual before first transformer block's attention.
- `_classify_param()` — add "engram" category for quantization dispatch.
- `mixed_quantize_int6()` — route engram embeddings to int4, projections to int6.

### Strategy 11C: Full Engram — Funded by Int5 MLP Savings (Combine with §8)

After §1B frees ~2.1MB via int5 MLP, allocate ~500KB to a sized-for-budget Engram table.

| What changes | Storage cost | Wallclock cost |
|---|---|---|
| 2-gram table: 10K slots × dim 16, 2 hash heads, int5 quantized | 10K × 16 × 5/8 × 2 = **~200KB** | Small |
| 3-gram table: 5K slots × dim 16, 1 hash head, int5 quantized | 5K × 16 × 5/8 = **~50KB** | Small |
| W_K + W_V projections (512 × 32) + conv | ~25KB at int6 | Negligible |
| **Total** | **~275KB** | **<2ms/step est.** |

*Note: embedding tables are expensive at scale. 100K slots × dim 64 × int5 = 4MB per head — far too large. The table above uses deliberately small dims (16) and moderate slot counts to stay within budget.*

**Buys:** Learned embeddings capture actual N-gram statistics, not just hash artifacts. Multi-head hashing reduces collisions. Even dim=16 embeddings can encode useful local context when projected through the gating mechanism.

This is the "Engram + 12th layer" play: int5 MLP frees ~2.1MB. Use ~0.3MB for Engram tables and ~1.8MB for 12th transformer layer (at int5). Net: same 16MB budget, but model gains both explicit memory AND an extra layer of depth.

Risks:
- This stacks three changes (int5 MLP + Engram + 12th layer), making attribution difficult.
- Dim=16 is much smaller than the paper's typical dim=1280. The gating projections (W_K, W_V) must bridge the dim gap — they project from dim 512 hidden state to the 32-dim embedding space (concatenation of 2-gram + 3-gram). Quality of the gating signal at this compression ratio is untested.
- 3-gram table at 5K slots covers only ~5K/1024^3 ≈ 0.0005% of possible 3-grams. Relies heavily on Zipfian distribution concentrating most probability mass in common patterns.
- Wallclock budget: 12th layer costs ~9% step time. Engram adds ~1-2ms. Total overhead ~12%, reducing max steps from ~7100 to ~6250. Must verify that depth + memory gain outweighs fewer training steps.

Mitigations:
- Test in sequence: (1) int5 MLP alone, (2) int5 + Engram alone, (3) int5 + 12th layer alone, (4) all three. This isolates each contribution.
- The paper's U-shaped law suggests optimal memory allocation is ~20-25% of spare budget. At 2.1MB freed, ~275KB (13%) is conservative — there's room to increase slot count or dim if initial results are positive.
- If 3-grams don't help (paper shows marginal benefit at small budgets), drop to 2-gram only and double the slot count to 20K.

**Recommended test sequence (integrates with existing Phase 2-4):**
1. Phase 2 (zero-cost): Test Strategy 11A (gated BigramHash) on existing checkpoint
2. Phase 3 (MLX ablation): Test Strategy 11B (micro-Engram) with small training runs
3. Phase 4 (full training): Strategy 11C combined with §8 (12th layer + int5 MLP + full Engram)

---

## 12. DeepCrossAttention: Learnable Residual Connections (arXiv:2502.06785)

**Paper:** Heddes et al. (Google Research, ICML 2025). "DeepCrossAttention: Supercharging Transformer Residual Connections." [[summary]](knowledge/summary_deep_cross_attention.md)

**Core insight:** Standard residual connections (x + f(x)) treat all prior layers equally, causing "information dilution." DCA replaces this with learnable, input-dependent weighted combinations of ALL previous layer outputs. Three independent GRNs compose Q, K, V inputs from the full prior-layer stack, enabling cross-depth interactions. A 30-layer DCA model outperforms a 42-layer standard transformer. Crucially, **DCA benefit is inversely proportional to model width** — our narrow dim=512 model is in the sweet spot.

**Why it fits Parameter Golf:** We already have `resid_mix` (2-channel x + x0 mixing) and `skip_weights` (U-Net encoder→decoder connections). DCA generalizes both into a single, more expressive mechanism. With only 11 layers, the full layer stack is small (no k-DCA truncation needed). Paper shows DCA reaches transformer quality in 1/3 training time — critical for our 600s wallclock budget.

**Parameter cost:** ~118K params (~88KB at int6). If DCA replaces `resid_mix` (~121K params saved), the net cost is approximately zero.

### Strategy 12A: Minimal DCA — GRN-v1 Only (Ablation First)

The paper's ablation shows GRN-v1 (dimension-independent scalar weights per layer) provides the single biggest improvement. This is the cheapest test.

| What changes | Param cost | Wallclock cost |
|---|---|---|
| Add scalar weight b_t per prior layer at each block's residual connection | T(T-1)/2 = 11×10/2 = **55 params total** | Negligible |
| Each block: x_in = sum(b_t * layer_output_t) instead of x + f(x) | — | — |

This is essentially DenseFormer applied to our architecture. Validates whether learnable residual weighting helps before adding the full DCA machinery.

Risks:
- May interact poorly with existing `resid_mix` and `skip_weights` — three overlapping residual mechanisms.
- 66 params is so few that quantization/precision is irrelevant; they should stay FP32.

Mitigations:
- Test with `resid_mix` removed (DCA subsumes it). Keep `skip_weights` initially.
- Zero-cost test: initialize b_t to match current ResNet behavior (all ones), train briefly on MLX.

### Strategy 12B: Full DCA — Replace resid_mix + skip_weights

Replace both `resid_mix` and `skip_weights` with three independent GRN-v3 instances per block (one each for Q, K, V inputs to attention).

| What changes | Param cost | Wallclock cost |
|---|---|---|
| Remove `resid_mix` from all 11 blocks | Saves **~121K params** | Saves ~negligible |
| Remove `skip_weights` (5 params × 512 dim) | Saves **~2.5K params** | Saves ~negligible |
| Add 3 × GRN-v3 per block (Q, K, V composition) | **~118K params** | ~2-5% step time (~2-4ms) |
| **Net** | **~0 params (roughly break-even)** | **~2-5% overhead** |

Each GRN-v3 at layer t:
- b_t: learned d × t bias matrix (dimension-dependent weights over prior layers)
- w_t: d-dim vector → ReLU(w^T * G_t) for input-dependent gating
- Output: (G_t ⊙ (b_t + w_t)) * 1 — weighted sum of prior layer outputs

**Buys:** Three independent views of the layer history for Q, K, V. The model can route Q to early-layer features, K to mid-layer, V to late-layer (or any other combination it learns). This is strictly more expressive than our current fixed U-Net + resid_mix pattern.

Risks:
- **torch.compile compatibility**: G_t stack has different sizes at each layer (1 at layer 0, 11 at layer 10). Options: (a) pad to max_layers at all layers, (b) use k=2 (fixed 4-entry stack). Padding wastes some compute; k=2 may lose some expressiveness.
- Removing U-Net skip connections loses an inductive bias that may be hard to relearn from scratch in 7100 steps. The skip connections encode a specific encoder→decoder pairing that took the community iterations to discover.
- 2-5% wallclock overhead reduces max steps by ~150-350. Must verify quality gain exceeds the cost of fewer steps.

Mitigations:
- **Keep U-Net skips as initialization**: Set the initial GRN weights to replicate the current U-Net + resid_mix behavior, then let training refine them. This preserves the inductive bias while unlocking more flexibility.
- For torch.compile, pre-allocate a (batch, max_layers, seq_len, dim) buffer and use slicing — avoids dynamic shapes.
- The paper shows DCA stabilizes training and eliminates loss spikes — this could offset some wallclock cost by enabling higher learning rates or fewer warmup steps.

**Code change locations:**
- New `GRN` class implementing GRN-v3 (input-dependent + dimension-dependent weights).
- New `DCABlock` replacing `Block` — contains attn_norm, mlp_norm, 3 GRN instances (for Q, K, V), attn, mlp. Removes resid_mix.
- `GPT.__init__()` — remove `skip_weights`, maintain a layer output buffer.
- `GPT.forward()` — accumulate layer outputs into a stack buffer, pass to each DCABlock.
- `CausalSelfAttention.forward()` — accept pre-composed Q, K, V inputs (currently computes Q/K/V internally via `self.qkv`). This requires splitting the QKV projection or having GRN operate on the pre-norm hidden state.
- `_classify_param()` — add "grn" category for DCA weights (keep FP32, do not quantize).
- **Alternative simpler approach**: Apply GRN only to the block input (not separately to Q/K/V). This avoids modifying CausalSelfAttention and captures most of the benefit (GRN-v3 alone is 18.41 PPL vs DCA's 18.06 in the paper's ablation).

### Strategy 12C: DCA + Engram + 12th Layer (The Full Stack)

Combine DCA (§12B) + Engram (§11C) + int5 MLP (§1B) + 12th layer (§8). The complete "depth-and-memory-and-routing" play.

| Component | Param cost | Artifact cost | Wallclock cost |
|---|---|---|---|
| Int5 MLP | 0 (same params, lower precision) | **Frees ~2.1MB** | 0 |
| DCA (replaces resid_mix + skip_weights) | ~0 net | ~0 net | +2-5% |
| Engram (§11C, 2+3 gram tables) | +new params | **~950KB** | +1-2% |
| 12th transformer layer | +2.4M params | **~1.1MB** | +9% |
| **Total** | — | **~16MB (within budget)** | **+12-16%** |

12-16% wallclock overhead reduces max steps from ~7100 to ~5950-6250. This must be offset by:
- DCA's 3x training efficiency claim (even 1.5x would suffice)
- Engram's effective depth increase (paper shows layer 5 ≈ layer 12)
- The actual 12th layer's contribution

This is the high-risk, high-reward play. Only attempt after each component is validated independently.

---

## 13. Other Unexplored Directions (Backlog)

### Evaluation
- Stride=32 instead of 64 (doubles eval time to ~146s, well within 600s budget)
- Adaptive stride (shorter stride for harder passages)
- Stride=32 + eval recurrence (~292s, still within budget)

### Architecture
- Mixture of tiny experts in MLP (route tokens to specialized sub-MLPs)
- XSA-all layers: promising signal from nanochat d12, but requires GQA-specific validation on our architecture (see §9)

### Quantization
- Full GPTQ with Hessian-based sensitivity weighting
- Learned quantization scales (train the scales, not just the weights)

### Training
- EMA decay scheduling (0.997 was carried forward without tuning)
- Longer warmdown (3500 → 4000?)
- Earlier QAT for int4/int5 tolerance

---

## Priority Order

Bug fixes are complete — see [Appendix: Completed Fixes](#appendix-completed-fixes).

### Phase 1 — Build quantization frontier harness
**Before any training runs, sweep quantization by component and layer on the existing SOTA checkpoint.** Record actual compressed bytes AND BPB for each configuration. This one tool validates or kills Strategies B, D, and E much faster than retraining.

Sweep axes:
- Per-component precision: MLP fc int5, MLP proj int5, MLP int4, attention int5, attention int8
- Per-layer precision: early/middle/late zones from Strategy E
- Measure: compressed artifact size (bytes after zstd-22) AND post-quantization BPB

This separates artifact-budget questions from wallclock-budget questions. Many ideas in this doc blend "frees MB" with "same 600s training" — the harness quantifies the artifact dimension independently.

### Phase 2 — Zero-cost tests (no training needed, run on existing model):
1. **Test stride=32 sliding window** — time it, measure BPB gain. Also validates eval budget headroom that recurrence depends on.
2. **Re-quantize with per-row GPTQ-lite** — measure BPB improvement from better quantization alone.
3. **Test int5 MLP quantization (§1 Strategy B)** — re-quantize existing model with int5 MLP, measure BPB delta.
4. **Test attention int8 / MLP int5 swap (§1 Strategy D)** — re-quantize with split precision, measure BPB. Note: int8 attn compression ratio (~1.25x) needs empirical validation.
5. **Test eval-only recurrence (§3 Approach A)** — run forward twice at eval, measure BPB.
6. **Test Engram-Lite gating (§11 Strategy A)** — add gating projections to BigramHash on existing checkpoint, measure BPB.

Note: the sliding eval bug (deferred) means all BPB numbers from these tests include a small tail-token bias. Keep raw NLL and token counts alongside BPB for recomputation.

### Phase 3 — Small-scale MLX ablations (quick training runs):
7. **XSA-all vs XSA-last-4 on our GQA architecture (§9)** — revalidate nanochat finding. GQA grouping may change the result.
8. **MLP:Attention rebalance (§7)** — head-to-head: 8Q/4KV 3.0xMLP vs 8Q/8KV 2.5xMLP.
9. **Remove resid_mix from decoder (§4)** — ablate alone before combining with other changes.
10. **Replace SmearGate with depthwise trigram conv (§5)** — ablate separately from resid_mix removal.
11. **Remove Value Embedding (§6)** — ~164K params. Ablate alone to isolate its contribution.
12. **Micro-Engram with learned embeddings (§11 Strategy B)** — small training run to validate learned 2-gram tables + gating at our scale.
13. **DCA GRN-v1 scalar weights (§12 Strategy A)** — cheapest DCA test: learnable scalar weights on residual connections, replace resid_mix.
14. **Full DCA replacing resid_mix + skip_weights (§12 Strategy B)** — if step 13 helps, upgrade to 3 × GRN-v3 for Q/K/V composition.

### Phase 4 — First full training runs:
15. **12th layer via int5 MLP (§8)** + best XSA config from step 7. The straightforward play.
16. **U-shaped precision curve (§1 Strategy E)** — int5 middle attention + int4 middle MLP fc. The novel quantization play.
17. **Full Engram + 12th layer (§11 Strategy C + §8)** — int5 MLP frees ~2.1MB; allocate ~1MB to Engram tables + ~1.1MB to 12th layer. The depth-and-memory play.
18. **DCA + Engram + 12th layer (§12 Strategy C)** — the full stack: DCA residuals + Engram memory + int5 MLP + 12th layer. High-risk, high-reward.

### Phase 5 — Combine winners:
19. **Best of 7-18** + weight-shared layers (§3 Approach B) — only if weight sharing still looks attractive after the earlier tests.
20. **Full stack:** best XSA + 12L + int5 MLP + DCA + Engram + U-shaped precision + eval recurrence + stride=32.

Not prioritized: §3 Approach C (train-time recurrence with curriculum) is high-risk and costs ~1200 training steps. Revisit only if eval-only recurrence (step 5) shows a clear win. §13 backlog items (MoE, full GPTQ, learned quant scales, EMA tuning, warmdown tuning) are parked until Phase 4 results are in.

---

## Appendix: Completed Fixes

*Source: our review + Codex's BUGS_AND_EFFICIENCY_REVIEW.md on the SOTA submission. All fixes applied to root `train_gpt.py` on 2026-03-24.*

| ID | Issue | Status | Summary |
|---|---|---|---|
| Bug 1 | Late QAT dead code (blocked int5/int4) | FIXED | Two-phase compile (`compiled_no_qat` + `compiled_qat`), symmetric clamp, `_clip_range` hook added for future per-module precision |
| Bug 2 | GPTQ-lite per-matrix not per-row | FIXED | `quantize_int6_per_row()` now tracks best percentile per row independently |
| Bug 3 | Sliding eval double-counts tail tokens | DEFERRED | Present in all submissions since 2026-03-19. Flagging to officials instead of fixing. |
| Bug 4 | SWA accumulated but never applied | FIXED | All SWA code deleted — hyperparams, state, accumulation loop |
| Eff 1 | Eval recompiles forward_logits each call | FIXED | `eval_val_sliding()` accepts `compiled_logits` param, compiled once at call site |
| Eff 2 | Data loader non-pinned CPU memory | FIXED | `DistributedTokenLoader` lazily allocates pinned staging buffers |
