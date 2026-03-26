#!/usr/bin/env python3
"""Quantization Frontier Sweep — test 6 quant configs on a raw fp32 checkpoint.

Usage:
    uv run experiments/quant_frontier.py [checkpoint_path]

Defaults to ../checkpoints/baseline_raw.pt if no path given.
Outputs JSON to experiments/results/quant_frontier.json
"""
from __future__ import annotations
import collections.abc
import glob
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import zstandard
except ImportError:
    raise ImportError("zstandard required: uv pip install zstandard")

from flash_attn_interface import flash_attn_func as flash_attn_3_func

# ─── Constants (must match baseline train_gpt.py) ───────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
    "smear", "dtg_gate", "ve_layer_scales", "ve_shared.scale",
)
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_PER_ROW_SCALE_DTYPE = torch.float16

# ─── Model hyperparameters (defaults matching SOTA baseline) ─────────────────
MODEL_DEFAULTS = dict(
    vocab_size=int(os.environ.get("VOCAB_SIZE", 1024)),
    num_layers=int(os.environ.get("NUM_LAYERS", 11)),
    model_dim=int(os.environ.get("MODEL_DIM", 512)),
    num_heads=int(os.environ.get("NUM_HEADS", 8)),
    num_kv_heads=int(os.environ.get("NUM_KV_HEADS", 4)),
    mlp_mult=float(os.environ.get("MLP_MULT", 3.0)),
    tie_embeddings=bool(int(os.environ.get("TIE_EMBEDDINGS", "1"))),
    tied_embed_init_std=float(os.environ.get("TIED_EMBED_INIT_STD", 0.005)),
    logit_softcap=float(os.environ.get("LOGIT_SOFTCAP", 30.0)),
    rope_base=float(os.environ.get("ROPE_BASE", 10000.0)),
    qk_gain_init=float(os.environ.get("QK_GAIN_INIT", 1.5)),
    bigram_vocab_size=int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048)),
    bigram_dim=int(os.environ.get("BIGRAM_DIM", 128)),
    xsa_last_n=int(os.environ.get("XSA_LAST_N", 4)),
    rope_dims=int(os.environ.get("ROPE_DIMS", 16)),
    ln_scale=bool(int(os.environ.get("LN_SCALE", "1"))),
    ve_enabled=bool(int(os.environ.get("VE_ENABLED", "1"))),
    ve_dim=int(os.environ.get("VE_DIM", 128)),
    ve_layers=os.environ.get("VE_LAYERS", "9,10"),
)
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
VAL_FILES = os.path.join(DATA_PATH, "fineweb_val_*.bin")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
EVAL_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", 2048))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", 64))


# ─── Model Classes (copied from baseline train_gpt.py) ──────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _clip_range: int = 31
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024,
                 rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2,
                                                  dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device,
                dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2,
                    dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor,
                     rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init,
                                               dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:],
                                           27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0,
                 ln_scale: bool = False, dtg: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                         qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(self, x: Tensor, x0: Tensor,
                v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                             v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = (x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :]
                 * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor))
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 mtp_num_heads: int = 0, mtp_loss_weight: float = 0.1,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 xsa_last_n: int = 0, rope_dims: int = 0, ln_scale: bool = False,
                 dtg: bool = False, ve_enabled: bool = False, ve_dim: int = 128,
                 ve_layers: str = "9,10"):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = (BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
                       if bigram_vocab_size > 0 else None)
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, layer_idx=i, ln_scale=ln_scale, dtg=dtg)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base,
                                           train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = ([int(x) for x in ve_layers.split(",") if x.strip()]
                                 if ve_enabled else [])
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32))
                 for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = (None if tie_embeddings
                        else CastedLinear(model_dim, vocab_size, bias=False))
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False)
             for _ in range(mtp_num_heads)])
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0,
                            std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2 and module.weight.shape[0] >= 64
                      and module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _get_ve(self, layer_idx: int, input_ids: Tensor,
                ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = (ve_cache['ve'] if ve_cache is not None
                   else self.ve_shared(input_ids))
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if ((param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS))
                    and param.dtype != torch.float32):
                param.data = param.data.float()


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if (sp.is_control(token_id) or sp.is_unknown(token_id)
                or sp.is_unused(token_id)):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_val(
    model: nn.Module, device: torch.device,
    val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    seq_len: int, val_batch_size: int,
) -> tuple[float, float]:
    local_batch_seqs = val_batch_size // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device,
                                                      dtype=torch.int64,
                                                      non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids]
                            & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    model: nn.Module, device: torch.device,
    val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    seq_len: int, stride: int, batch_seqs: int = 32,
) -> tuple[float, float]:
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt]
                       & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    return val_loss, bits_per_token * tokens_per_byte


# ─── Quantization ────────────────────────────────────────────────────────────

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def _extract_layer_idx(name: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        nrows = t32.shape[0]
        best_err = torch.full((nrows,), float('inf'), device=t32.device)
        best_q = torch.zeros_like(t32, dtype=torch.int8)
        best_s = torch.zeros(nrows, dtype=torch.float16, device=t32.device)
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]),
                            -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            row_err = (t32 - recon).pow(2).mean(dim=1)
            improved = row_err < best_err
            best_q[improved] = q[improved]
            best_s[improved] = s[improved]
            best_err[improved] = row_err[improved]
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0,
                         dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()),
                    -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """Int8 quantization fallback for embed/other params."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel()
                    else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]),
                                -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]),
                        -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = (float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
                if t32.numel() else 0.0)
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0,
                         dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
        -127, 127).to(torch.int8).contiguous()
    return q, scale


# ─── U-shaped precision curve (from IDEAS.md §1E) ───────────────────────────

def _get_clip_range_ushaped(name: str, layer_idx: int | None, cat: str) -> int:
    if layer_idx is None:
        return 31
    if 3 <= layer_idx <= 7:  # middle zone
        if cat == "attn":
            return 15   # int5 attention
        if cat == "mlp" and ".fc." in name:
            return 7    # int4 MLP fc
        if cat == "mlp":
            return 15   # int5 MLP proj
    if layer_idx >= 8:  # late zone
        if cat == "attn":
            return 127  # int8 attention
        if cat == "mlp" and ".proj." in name:
            return 31   # int6 MLP proj
        if cat == "mlp":
            return 15   # int5 MLP fc
    # early zone (layers 0-2): int6 attn, int5 MLP
    return 31 if cat == "attn" else 15


# ─── Config-aware quantize/dequantize ────────────────────────────────────────

def mixed_quantize_configurable(
    state_dict: dict[str, Tensor],
    clip_range_fn: collections.abc.Callable[[str, str], int | None],
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Quantize with per-param clip ranges.

    clip_range_fn(param_name, category) -> clip_range or None.
    None means use int8 float quantization (for embed/other params).
    """
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        cr = clip_range_fn(name, cat)
        if cr is not None and t.ndim >= 1:
            q, s = quantize_int6_per_row(t, clip_range=cr)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed(
    result: dict[str, Tensor],
    meta: dict[str, object],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Reconstruct fp32 tensors from quantized representation."""
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if (t.dtype == torch.float16
                    and orig_dtype in (torch.float32, torch.bfloat16)):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float()
                         * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
                         ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


# ─── Quantization Configs ────────────────────────────────────────────────────

def _make_uniform_fn(mlp_clip: int, attn_clip: int):
    def fn(name: str, cat: str) -> int | None:
        if cat == "mlp":
            return mlp_clip
        if cat == "attn":
            return attn_clip
        return None
    return fn


def _make_split_mlp_fn(mlp_fc_clip: int, mlp_proj_clip: int, attn_clip: int):
    def fn(name: str, cat: str) -> int | None:
        if cat == "attn":
            return attn_clip
        if cat == "mlp":
            return mlp_fc_clip if ".fc." in name else mlp_proj_clip
        return None
    return fn


def _ushaped_fn(name: str, cat: str) -> int | None:
    if cat not in ("mlp", "attn"):
        return None
    layer_idx = _extract_layer_idx(name)
    return _get_clip_range_ushaped(name, layer_idx, cat)


CONFIGS: list[tuple[str, collections.abc.Callable]] = [
    ("int6_uniform",  _make_uniform_fn(31, 31)),
    ("int5_mlp_fc",   _make_split_mlp_fn(15, 31, 31)),
    ("int5_mlp_all",  _make_uniform_fn(15, 31)),
    ("attn8_mlp5",    _make_uniform_fn(15, 127)),
    ("attn8_mlp4",    _make_split_mlp_fn(7, 15, 127)),
    ("ushaped",       _ushaped_fn),
]


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "../checkpoints/baseline_raw.pt"
    ckpt_path = str(Path(ckpt_path).resolve())
    print(f"Loading checkpoint: {ckpt_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load checkpoint (raw fp32 state dict)
    sd_cpu = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded {sum(t.numel() for t in sd_cpu.values())} params")

    # Build model and load original weights
    model_cfg = {k: v for k, v in MODEL_DEFAULTS.items()}
    model_cfg["mtp_num_heads"] = 0
    model_cfg["mtp_loss_weight"] = 0.0
    eval_model = GPT(**model_cfg).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(sd_cpu, strict=True)
    print("Model loaded successfully")

    # Load validation data
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(VAL_FILES, EVAL_SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, MODEL_DEFAULTS["vocab_size"], device)
    print(f"Validation tokens: {val_tokens.numel() - 1}")

    # Reference state dict for dequantization (keeps original dtypes)
    template_sd = {k: v.detach().cpu() for k, v in sd_cpu.items()}

    results: list[dict] = []
    for config_name, clip_range_fn in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        t0 = time.perf_counter()

        # Quantize
        quant_result, quant_meta = mixed_quantize_configurable(
            template_sd, clip_range_fn)

        # Compress with zstd-22
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        compressed_bytes = len(quant_blob)
        print(f"  Compressed: {compressed_bytes:,} bytes "
              f"({compressed_bytes / 1e6:.2f} MB)")

        # Decompress + dequantize
        decompressed = zstandard.ZstdDecompressor().decompress(quant_blob)
        quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
        deq_state = dequantize_mixed(
            quant_state["w"], quant_state["m"], template_sd)

        # Load into eval model
        eval_model.load_state_dict(deq_state, strict=True)

        # Eval: non-sliding BPB
        torch.cuda.synchronize()
        t_eval = time.perf_counter()
        _, val_bpb_nonsliding = eval_val(
            eval_model, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            seq_len=EVAL_SEQ_LEN, val_batch_size=VAL_BATCH_SIZE,
        )
        torch.cuda.synchronize()
        print(f"  Non-sliding BPB: {val_bpb_nonsliding:.4f} "
              f"({time.perf_counter() - t_eval:.1f}s)")

        # Eval: sliding window BPB
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        _, val_bpb_sliding = eval_val_sliding(
            eval_model, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            seq_len=EVAL_SEQ_LEN, stride=EVAL_STRIDE,
        )
        torch.cuda.synchronize()
        print(f"  Sliding BPB:     {val_bpb_sliding:.4f} "
              f"({time.perf_counter() - t_slide:.1f}s)")

        elapsed = time.perf_counter() - t0
        print(f"  Total time: {elapsed:.1f}s")

        results.append({
            "config": config_name,
            "compressed_bytes": compressed_bytes,
            "val_bpb_nonsliding": round(val_bpb_nonsliding, 6),
            "val_bpb_sliding": round(val_bpb_sliding, 6),
        })

    # Write results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "quant_frontier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
