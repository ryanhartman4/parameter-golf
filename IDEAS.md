# Parameter Golf — Ideas for Beating 1.0810 BPB

Current SOTA: **1.0810 BPB** (bigbag, April 9, 2026)
Architecture: 11L × 512d × 8Q/4KV, 4× MLP, LeakyReLU(0.5)², SP8192, Partial RoPE (16/64),
layerwise LN scale, tied embeddings, logit softcap=30

Key techniques in the current record:
- **SP8192 tokenizer** — 8× larger vocab than baseline 1024
- **Depth recurrence** — layers 3-5 looped 2× (17 virtual layers from 11 physical)
- **Parallel residuals** — GPT-J style from layer 7+ (attn + mlp read same input)
- **Skip gates** — sigmoid-gated U-Net encoder→decoder connections
- **XSA all layers** — extended self-attention on all 11 layers
- **Full Hessian GPTQ** with SDClip (k=12.85 for matrices, k=20 for embeddings)
- **TTT** — score-first test-time training (SGD 3 epochs/chunk, cosine LR decay)
- **EMA** — decay 0.9965
- **Byte-shuffle + Brotli-11** compression
- **MuonEq-R** — row-normalized Muon, Newton-Schulz 5 steps
- Hyperparameters: WD=0.095, MLR=0.022, warmdown_frac=0.72

### Two Independent Bottlenecks

| Constraint | Budget | What helps | What hurts |
|---|---|---|---|
| **Artifact size** | 16 MB (code + compressed model) | Lower precision, better compression, fewer params | More layers, wider model, higher precision |
| **Wallclock** | 600s training + 600s eval on 8xH100 | Faster steps, eval-time compute (TTT) | More layers per step, heavier ops |

Training uses ~588s. Eval (sliding + TTT) uses ~500s. About **100s of eval headroom** remains.

---

## 1. Deeper Recurrence

The current record loops layers 3-5 twice (2 extra passes). Open questions:

### 1A: Three Loops
Add a third loop: `num_loops=3`. Gives 20 virtual layers from 11 physical. Cost: ~15% more step time, losing ~700 training steps. Worth it if effective depth >> raw step count.

### 1B: Wider Loop Range
Loop layers 2-6 instead of 3-5. More layers per loop = more effective depth per pass but heavier per-step cost. Also changes the encoder/decoder split point.

### 1C: Asymmetric Loops
Loop different layers different numbers of times. E.g., loop layers 4-5 three times but layer 3 only once. Requires modifying the index-building logic in `GPT.__init__`.

---

## 2. Quantization Precision with Hessian GPTQ

The current record uses uniform int6 for all weight matrices and int8 for embeddings. The SDClip formula (`scale = k * std(row) / clip_range`) provides a principled starting point for precision experiments.

### 2A: Per-Layer SDClip k-Values
Instead of a single `matrix_clip_sigmas=12.85` for all layers, tune k per layer or per component. Middle layers may tolerate lower k (more aggressive clipping); late layers may want higher k.

### 2B: Mixed Int5/Int6 with SDClip
Drop MLP weights to int5 (`clip_range=15`) while keeping attention at int6 (`clip_range=31`). With Hessian-based error compensation, int5 MLP may lose less quality than the old percentile-based approach.

**Buys:** ~2.1 MB freed → enough for a 12th layer or wider model.

### 2C: Int4 Middle Layers
U-shaped precision: int4 for MLP fc in middle layers (3-7), int6 attention everywhere, int8 late attention. The Hessian GPTQ should compensate for int4 quantization noise much better than our old GPTQ-lite.

---

## 3. TTT Improvements

TTT provides ~0.002 BPP improvement (1.0827 sliding → 1.0810 TTT) using ~370s of the 600s eval budget.

### 3A: More Epochs
Increase `ttt_epochs` from 3 to 5 or 8. Cost: more eval time. The cosine LR decay already accounts for longer training. Check if diminishing returns kick in.

### 3B: Different Optimizer
Replace SGD with AdamW or Muon for TTT. SGD with momentum is simple but may under-adapt. AdamW could converge faster per epoch, allowing more adaptation per time unit.

### 3C: Selective Parameter TTT
Only adapt MLP weights or only adapt the last few layers during TTT. Reduces parameter count for SGD, allowing higher LR without instability.

### 3D: Larger Chunks
Increase `ttt_chunk_tokens` from 32K to 64K. Each chunk gets more context for adaptation but fewer total adaptation steps.

---

## 4. Eval-Time Budget Analysis

| Phase | Time Used | Budget | Headroom |
|---|---|---|---|
| Training | ~588s | 600s | ~12s |
| Eval (sliding + TTT) | ~500s | 600s | **~100s** |

100s of eval headroom could fund:
- Eval-only depth recurrence (extra forward passes after training)
- Ensemble scoring (run model twice with different layer configs, average logits)
- Adaptive stride (stride=32 for hard passages, stride=128 for easy ones)

---

## 5. Engram: Conditional Memory (arXiv:2601.07372)

Still novel — nobody in the competition uses explicit N-gram memory tables.

The SP8192 vocab partially captures bigram/trigram statistics through larger tokens, but explicit learned memory tables could still help for rare or ambiguous patterns.

### Updated Strategy for SP8192 Baseline

The old BigramHashEmbedding used XOR hashing into a fixed table. With SP8192, the token space is 8× larger, so collision rates would be 64× higher for the same table size.

**Revised approach:** Use Engram-style multi-head hashing with prime-sized tables, gated injection. At int5, a 10K-slot × dim-16 table costs ~100KB — well within budget if int5 MLP savings are available.

---

## 6. DCA: Learnable Residual Connections (arXiv:2502.06785)

The current record already uses skip gates (sigmoid-gated U-Net connections), which provide some of DCA's benefit. The question is whether full DCA (learnable input-dependent weighted combinations of ALL prior layer outputs) adds enough over skip gates to justify the complexity.

### What Skip Gates Already Do
Skip gates provide a learned weighting between the encoder output and the current decoder hidden state at each decoder layer. This is a 2-way mix per layer.

### What DCA Adds
DCA composes Q, K, V inputs from the full prior-layer stack (all 11+ layers), not just the paired encoder layer. Three independent GRN instances per block. The paper shows DCA benefit is inversely proportional to model width — our narrow dim=512 model is in the sweet spot.

### Test Sequence
1. GRN-v1 scalar weights only (55 params total) — cheapest test of whether learnable residual weighting helps beyond skip gates
2. Full GRN-v3 replacing skip_weights + skip_gates — if step 1 helps

---

## 7. Architecture Exploration

### 7A: Wider Model (dim=640)
With better compression (Brotli + int5 MLP), there may be room for a wider model. dim=640 gives head_dim=80, 10 KV heads possible (640/64=10). Cost: ~56% more params, ~25% slower steps.

### 7B: 12th Layer
If int5 MLP frees ~2.1 MB, a 12th layer costs ~1.8 MB at int5. The U-Net becomes perfectly symmetric (6/6 instead of 5/6). Every decoder layer gets a skip connection.

### 7C: Vocabulary Expansion
SP16384 or SP32768. Larger vocab = shorter sequences = more context per training step. But: larger embedding table, harder to fit under 16MB.

---

## 8. Compression Improvements

The current record uses byte-shuffle(stride=2) + Brotli-11. Potential improvements:

- **Higher shuffle stride** (3 or 4) — may improve compression for int6 packed data
- **Brotli vs LZMA comparison** — the code supports both; measure which gives smaller artifacts
- **Separate compression for embeddings vs weights** — embeddings (int8) and weights (int6) may compress better with different strategies

---

## 9. QAT (Quantization-Aware Training)

No record submission uses QAT with the Hessian GPTQ quantizer. Our old QAT infrastructure (two-phase compile, symmetric clamp) needs redesign for the new GPTQ, but the concept is still valuable.

**Approach:** During the warmdown phase, inject fake-quantization noise matching the SDClip formula into the forward pass. The model learns to be robust to the specific quantization it will receive at export.

**Risk:** QAT adds complexity to the compile graph and may slow training. With only ~4550 steps total, every step counts.

---

## Priority Order

### Phase 1 — Zero-cost tests (existing checkpoint, no training):
1. Test different SDClip k-values per layer (sweep matrix_clip_sigmas)
2. Test int5 MLP with Hessian GPTQ (measure BPB delta vs int6)
3. Test higher byte-shuffle stride (3, 4) for compression
4. Test TTT with more epochs (5, 8)
5. Test eval-only depth recurrence (extra forward pass)

### Phase 2 — Small ablations (short training runs):
6. Three loops vs two loops
7. Wider loop range (layers 2-6)
8. TTT with AdamW or selective parameters
9. Engram-lite (gated hash table injection)
10. DCA GRN-v1 scalar weights

### Phase 3 — Full training runs:
11. Int5 MLP + 12th layer
12. Mixed precision (layerwise SDClip k-values + int5 MLP)
13. Full Engram with learned tables
14. Full DCA replacing skip gates

### Phase 4 — Combine winners:
15. Best quantization + best architecture + best TTT config
