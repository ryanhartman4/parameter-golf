# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

**Paper:** Cheng, Zeng, Dai, Chen, Wang, et al. (2025). "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models." arXiv:2601.07372
**Code:** https://github.com/deepseek-ai/Engram

---

## Core Idea

Transformers lack a native knowledge lookup primitive and waste early layers reconstructing static patterns (named entities, formulaic phrases, N-gram regularities) through expensive computation. Engram introduces "conditional memory" as a complementary sparsity axis to MoE: instead of routing through expert networks, it performs O(1) hash-based lookups into massive embedding tables to retrieve static N-gram embeddings, then fuses them into the residual stream via context-aware gating.

The key insight is a **U-shaped scaling law for sparsity allocation**: given a fixed total parameter budget, the optimal split devotes ~75-80% to MoE experts and ~20-25% to Engram memory. Pure MoE is suboptimal because it forces the model to "simulate retrieval through computation" in early layers.

## Method

**Architecture:**
1. **Tokenizer Compression**: Collapse raw token IDs into canonical IDs via NFKC normalization + lowercasing (~23% vocab reduction for 128K tokenizer)
2. **Multi-Head Hashing**: For each N-gram order (2-gram, 3-gram), use K distinct hash heads mapping to embedding tables of prime size. Hash function is multiplicative-XOR. Final memory vector is the concatenation of all retrieved embeddings across N-gram orders and hash heads.
3. **Context-Aware Gating**: Hidden state h_t serves as Query, retrieved embedding e_t provides Key and Value. Scalar gate alpha_t = sigmoid(RMSNorm(h_t)^T * RMSNorm(W_K * e_t) / sqrt(d)). Gate suppresses noise from hash collisions or irrelevant retrievals.
4. **Depthwise Causal Convolution**: Kernel size 4, dilation = max N-gram order, SiLU activation, applied after gating.
5. **Residual injection**: H^(l) += Y (gated + conv output), placed before standard Attention + FFN/MoE.

**Placement:** Layer 2 is optimal for single insertion (one attention layer provides enough context for gating). Splitting memory across layers 2 and 6 (or 2 and 15 at scale) works even better.

**Training:** Embedding params use Adam with 5x learning rate and no weight decay. Conv params initialized to zero (identity mapping at start). Standard Muon optimizer for backbone.

**System efficiency:** Deterministic hash IDs enable prefetching from host memory during inference. 100B-parameter table offloaded to CPU incurs <3% throughput penalty because preceding transformer blocks provide compute time to hide PCIe latency.

## Key Findings

1. **U-shaped allocation law**: At both 5.7B and 9.9B scale, optimal rho ~75-80% (MoE) / ~20-25% (Engram). Pure MoE (rho=100%) is strictly suboptimal.
2. **Engram-27B vs MoE-27B** (iso-param, iso-FLOPs): Engram wins across ALL categories — knowledge (+3.0 MMLU), reasoning (+5.0 BBH), code (+3.0 HumanEval), math (+2.4 MATH).
3. **Effective depth increase**: CKA analysis shows Engram layer 5 representations align with MoE layer 12. LogitLens shows faster prediction convergence in early layers. Engram "deepens" the network by offloading static reconstruction.
4. **Layer placement ablation**: Layer 2 optimal for single insertion. Performance degrades monotonically as insertion moves deeper. Two insertions (layers 2+6) beat single insertion.
5. **Component ablation** (most to least important): Multi-branch integration > context-aware gating > tokenizer compression > depthwise conv (marginal). 4-grams slightly hurt at 1.6B budget (dilute 2/3-gram capacity).
6. **Infinite memory scaling**: Log-linear improvement in val loss as embedding slots increase from 258K to 10M. Engram provides a predictable, compute-free scaling knob.
7. **Sensitivity**: Disabling Engram at inference collapses factual knowledge (29-44% retained) but barely affects reading comprehension (81-93% retained).

---

## Relevance to This Project

### The core opportunity: Engram as a free depth booster for early layers

Parameter Golf's central challenge is L(N) optimization under a 16MB artifact + 10-minute wallclock budget. The current 11-layer model spends early layers reconstructing local patterns — exactly the waste Engram is designed to eliminate. The paper's CKA analysis directly parallels our situation: if Engram can make layer 2 "act like" layer 5, we effectively get 2-3 free layers of depth without adding transformer blocks (which cost ~9% wallclock each).

### Why it could work here

1. **We already have the infrastructure.** Our `BigramHashEmbedding` (lines ~550-590 in train_gpt.py) is essentially a simplified Engram without the gating or multi-head hashing. It uses XOR-based hashing into an embedding table and learns a scale factor. Engram's multi-head hash + context-aware gating is a strict upgrade.

2. **Our vocab is tiny (1024 tokens).** With 1024 vocab, 2-gram space is only 1M combinations and 3-gram space is ~1B. Hash tables with prime-sized buckets can cover this efficiently. Tokenizer compression is irrelevant (our vocab is already tiny), but the small vocab means hash collisions are far less frequent than with 128K tokenizers.

3. **We're desperate for effective depth.** Adding a 12th transformer layer costs ~9% wallclock and ~2.4M params. If Engram can provide equivalent depth gain at a fraction of the parameter cost and near-zero FLOP cost, it's strictly better.

4. **Placement at layer 2 aligns with our architecture.** The paper confirms one attention layer is sufficient context for gating. Our U-Net architecture (skip connections) already has an asymmetric compute profile — injecting Engram in the early encoder layers would complement the skip-connection depth advantage in later layers.

### Concrete parameter budget analysis

Current budget: 15.5MB compressed (int6 + zstd-22). ~0.5MB headroom to 16MB.

A minimal Engram module would need:
- **Hash tables**: With 1024 vocab, 2-gram needs ~1M slots, 3-gram needs ~1B slots (too big). Even modest tables are expensive: 10K slots × dim 64 × int6 = 10K × 64 × 6/8 = ~480KB per head. **Two heads = ~960KB — nearly the entire headroom.**
- **Smaller approach**: 2-gram only, 2 hash heads, 3K slots, dim 16 = 3K × 16 × 5/8 × 2 = ~60KB. Feasible but very small.
- **Gating projections**: W_K (dim 512 × 16 = 8K params) + W_V (same) = ~16K params = ~12KB at int6. Negligible.

### Critical constraint: artifact size is the blocker

The paper operates at billions of parameters for Engram tables, but we have 0.5MB of headroom. This is the fundamental tension. Our existing BigramHashEmbedding uses ~0.3-0.5MB and only learns a scale factor (table is computed at runtime via XOR). Engram's power comes from *learned* embeddings, which must be stored.

### Viable adaptations for Parameter Golf

1. **Micro-Engram with aggressive quantization**: 3K-5K slots, 2-gram only, dim 16-32, int4 quantization. At int4: 3K × 16 × 0.5 × 2 heads = ~48KB. Feasible within budget but extremely compressed. The question is whether such a tiny table provides meaningful signal.

2. **Hybrid: keep XOR hash table + add learned gating**: Instead of storing full learned embeddings, keep the existing BigramHash approach (runtime-computed table, zero storage cost) but add the context-aware gating mechanism. The gating projection matrices (W_K, W_V) are small (~48KB at int6). This captures the "suppress noise" benefit without the storage cost of learned embeddings.

3. **Engram-inspired depth recurrence**: Use the insight that Engram "deepens" early layers to inform our depth recurrence strategy (IDEAS.md section 3). If we can simulate the Engram effect through eval-time recurrence at early layers, we get the depth benefit at zero parameter cost.

4. **Replace BigramHashEmbedding entirely**: The existing BigramHash learns only a scale per position. A micro-Engram with multi-head hashing and gating could be more expressive at similar or slightly larger parameter count, while also providing the early-layer depth benefit.

### Open questions specific to this project

- At 1024 vocab, is 2-gram coverage dense enough that hash collisions are rare even with small tables? (1024^2 = ~1M 2-grams, so a 50K-slot table has ~20x collision rate)
- Does the context-aware gating mechanism help enough at our scale (512 dim, 11 layers) to justify the extra forward pass cost?
- Can we quantize Engram tables to int4 without destroying the gating signal?
- Would the wallclock cost of the hash + lookup + gating offset the depth benefit at our 84.5ms/step budget?
- Should this be combined with or replace the existing BigramHash + SmearGate modules?

### Recommendation

**Priority: Medium-Low for artifact-constrained track. High if combined with int5 MLP savings.**

The most promising approach is (2): keep the zero-storage XOR hash table but add learned gating (W_K, W_V projections ~48KB). This is essentially "Engram-lite" — the gating mechanism alone captures much of the benefit by letting the model suppress irrelevant N-gram signal, and the projections are cheap enough to fit in the artifact budget. Combine this with the int5 MLP savings (~2.1MB freed) and there may be room for a small learned embedding table on top.

Test this as a zero-cost ablation: modify BigramHashEmbedding to include gating, measure BPB on an existing checkpoint.
