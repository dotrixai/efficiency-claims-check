"""ECC-004 runner: claim C12 (nGPT, arXiv 2410.01131) against a modern GPT baseline on one GPU (Kaggle T4 or P100), at context 1024 and 4096.

One file holds both models, the training loop, the validity checks and the measurements. A run is reproduced from this file plus the token
files (see prepare_data.py).

    python ecc004.py train  --arm baseline --seed 11 --ctx 1024 --lr 2e-3 --tokens 5e8 --data-dir DIR --out DIR [--resume DIR/ckpt.pt]
    python ecc004.py checks --arm ngpt --ctx 1024 --vocab 8192            # causality, batch coupling and fp16 forward checks on a fresh model
    python ecc004.py flops  --arm ngpt --ctx 4096 --vocab 8192            # FLOPs per token, analytic and torch.utils.flop_counter

Both arms use the same shapes: d, layers, heads, a SwiGLU MLP with hidden size 4d (as in the paper), RoPE, untied input and output embeddings.

  baseline  pre-LayerNorm GPT (ECC baseline v1, scaled up) with QK-norm (per-head RMSNorm of q and k before RoPE, learnable gain):
            AdamW (0.9, 0.95), weight decay 0.1 on matrices except the input embedding,
            init std 0.02 with residual projections scaled by 1 / sqrt(2 layers), linear warmup then cosine.
  ngpt      the paper's normalized Transformer, checked against github.com/NVIDIA/ngpt (commit ed19eb2):
              - every hidden state on the unit sphere: h <- Norm(Norm(h) + |alpha| (Norm(h_X) - Norm(h))) after attention and after the MLP;
              - q and k normalized after RoPE, times s_qk (one vector per head), softmax scale sqrt(d_head);
              - SwiGLU input times s_uv * sqrt(d) (the code scales u and v; the paper only nu; u's scale cancels in the next Norm);
              - logits times s_z; no LayerNorm, no weight decay, no warmup;
              - all matrices renormalized along the embedding dimension at init and after every optimizer step;
              - scaling vectors stored with the paper's (init / scale) restoration: alpha (0.05, 1/sqrt(d)), s_qk (1, 1/sqrt(d)),
                s_uv (1, 1), s_z (1, 1/sqrt(d)).

Precision: parameters and optimizer state stay in fp32 on both arms (mixed precision through autocast, fp16 with a GradScaler on T4).
Norms, residual mixing and the loss run in fp32. This matters: NVIDIA's reimplementation notes that bf16 parameters hurt the GPT baseline
more than nGPT, which inflates the speed-up.

Measurement: train_time_s is GPU time of the training updates only (synchronized wall clock around each update; evaluation, checkpoints
and batch assembly excluded). FLOPs are counted analytically per token for one forward and backward pass (3x forward matmuls, attention
counted at full T x T like torch.utils.flop_counter does); the flop_counter value is logged as a cross-check. nGPT's normalizations are
not matmuls, so they appear in time only.

Compilation: with --compile inductor, the forward pass and the loss are compiled with torch.compile for BOTH arms (this fuses nGPT's
normalizations, which run as separate kernels in eager mode). Compilation happens in the first updates and is counted as training time.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclasses.dataclass
class Config:
    arm: str = 'baseline'
    d: int = 384
    layers: int = 8
    heads: int = 6
    hidden: int = 1536                    # SwiGLU hidden size (paper: 4d)
    qk_norm: int = 1                      # baseline only: per-head RMSNorm of q and k (a common modern addition; PREREG decision 4b)
    vocab: int = 8192
    ctx: int = 1024
    tokens: float = 5e8                   # training tokens of this run; the LR schedule spans exactly this budget
    tokens_per_update: int = 131072       # 2^17
    micro_rows: int = 16                  # rows per forward pass (gradient accumulation fills tokens_per_update)
    lr: float = 2e-3
    warmup_frac: float = -1.0             # -1: arm default (baseline 0.01, ngpt 0)
    min_lr_frac: float = 0.0              # cosine to 0, as in the paper (both arms)
    wd: float = -1.0                      # -1: arm default (baseline 0.1, ngpt 0)
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0                # both arms, as in NVIDIA/ngpt train.py
    amp: str = 'fp16'                     # fp16 | bf16 | fp32
    compile: str = 'none'                 # none | inductor (torch.compile of forward + loss, both arms) | aot_eager (CPU tests of the same path)
    seed: int = 11
    eval_tokens: int = 2 ** 21            # scored targets per evaluation
    eval_split: str = 'sealed'            # sealed (Q and the verdict) | tune (learning-rate tuning only)
    eval_fracs: tuple = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                         0.8, 0.85, 0.9, 0.95, 1.0)
    log_every: int = 25

    def __post_init__(self):
        assert self.arm in ('baseline', 'ngpt') and self.d % self.heads == 0 and self.tokens_per_update % self.ctx == 0
        if self.warmup_frac < 0:
            self.warmup_frac = 0.01 if self.arm == 'baseline' else 0.0
        if self.wd < 0:
            self.wd = 0.1 if self.arm == 'baseline' else 0.0
        self.betas, self.eval_fracs = tuple(self.betas), tuple(self.eval_fracs)

    @property
    def rows(self):
        return self.tokens_per_update // self.ctx

    @property
    def updates(self):
        return max(1, int(round(self.tokens / self.tokens_per_update)))


# ---------------------------------------------------------------------------------------------------------------------------- models

_ROPE = {}


def rope_tables(T, dh, device):
    key = (T, dh, str(device))
    if key not in _ROPE:
        half = dh // 2
        inv = 10000.0 ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half)
        ang = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None, :]
        _ROPE[key] = (torch.cos(ang), torch.sin(ang))
    return _ROPE[key]


def rope(x, cos, sin):
    """x [B, H, T, dh]; rotary embedding on (first half, second half) pairs, computed in fp32, returned in x's dtype."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


def norm(x, dim=-1):
    """L2 normalization in fp32 (returns fp32)."""
    x = x.float()
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def rms_norm(x, gain, eps=1e-6):
    """RMSNorm over the last dimension in fp32, times a learnable gain; returned in x's dtype (baseline QK-norm)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gain).to(x.dtype)


class Scale(nn.Module):
    """nGPT scaling vector with the paper's restoration: effective = param * (init / scale), param initialized to scale."""
    def __init__(self, n, init, scale):
        super().__init__()
        self.ratio = init / scale
        self.param = nn.Parameter(torch.full((n,), float(scale)))

    def forward(self):
        return self.param * self.ratio


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, self.dh = c, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)       # rows live in R^d (q, k, v stacked)
        self.o = nn.Linear(c.d, c.d, bias=False)             # columns live in R^d
        self.fc = nn.Linear(c.d, 2 * c.hidden, bias=False)   # u and the gate, stacked
        self.down = nn.Linear(c.hidden, c.d, bias=False)
        if c.arm == 'baseline':
            self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
            if c.qk_norm:                                   # per-head RMSNorm of q and k before RoPE, one gain vector shared by the heads
                self.q_gain, self.k_gain = nn.Parameter(torch.ones(self.dh)), nn.Parameter(torch.ones(self.dh))
        else:
            s = 1 / math.sqrt(c.d)
            self.alpha_a, self.alpha_m = Scale(c.d, 0.05, s), Scale(c.d, 0.05, s)
            self.s_qk = Scale(c.d, 1.0, s)
            self.s_uv = Scale(2 * c.hidden, 1.0, 1.0)

    def attend(self, a, B, T):
        q, k, v = self.qkv(a).view(B, T, 3, self.c.heads, self.dh).permute(2, 0, 3, 1, 4)   # each [B, H, T, dh]
        cos, sin = rope_tables(T, self.dh, a.device)
        if self.c.arm == 'baseline' and self.c.qk_norm:
            q, k = rms_norm(q, self.q_gain), rms_norm(k, self.k_gain)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if self.c.arm == 'baseline':
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            sqk = self.s_qk().view(1, self.c.heads, 1, self.dh)
            q, k = sqk * norm(q), sqk * norm(k)
            q, k, v = (t.to(v.dtype) for t in (q, k, v))     # attention in the autocast dtype; logits are bounded by s_qk^2 * sqrt(dh)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=math.sqrt(self.dh))
        return self.o(y.transpose(1, 2).reshape(B, T, self.c.d))

    def forward(self, h):
        B, T, _ = h.shape
        if self.c.arm == 'baseline':
            h = h + self.attend(self.ln1(h), B, T)
            u, g = self.fc(self.ln2(h)).chunk(2, -1)
            return h + self.down(u * F.silu(g))
        a = norm(h)
        h = norm(a + self.alpha_a().abs() * (norm(self.attend(h, B, T)) - a))
        uv = self.fc(h) * (self.s_uv() * math.sqrt(self.c.d))
        u, g = uv.chunk(2, -1)
        a = norm(h)
        return norm(a + self.alpha_m().abs() * (norm(self.down(u * F.silu(g))) - a))


class LM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        std = 0.02 if c.arm == 'baseline' else 1 / math.sqrt(c.d)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=std / math.sqrt(2 * c.layers) if n.endswith(('o.weight', 'down.weight')) else std)
        if c.arm == 'baseline':
            self.ln = nn.LayerNorm(c.d)
        else:
            self.s_z = Scale(c.vocab, 1.0, 1 / math.sqrt(c.d))
            self.renormalize()

    @torch.no_grad()
    def renormalize(self):
        """nGPT: unit norm along the embedding dimension for every matrix (rows for d -> *, columns for * -> d)."""
        for w in (self.emb.weight, self.head.weight):
            w.copy_(norm(w, 1))
        for b in self.blocks:
            for w in (b.qkv.weight, b.fc.weight):
                w.copy_(norm(w, 1))
            for w in (b.o.weight, b.down.weight):
                w.copy_(norm(w, 0))

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        if self.c.arm == 'baseline':
            return self.head(self.ln(h))
        return self.head(h) * self.s_z()


def make_optimizer(model, c, device):
    fused = dict(fused=True) if device.type == 'cuda' else {}
    if c.arm == 'baseline':
        decay = [p for n, p in model.named_parameters() if p.dim() == 2 and not n.startswith('emb')]
        rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and not n.startswith('emb'))]
        groups = [dict(params=decay, weight_decay=c.wd), dict(params=rest, weight_decay=0.0)]
    else:
        groups = [dict(params=list(model.parameters()), weight_decay=c.wd)]
    return torch.optim.AdamW(groups, lr=c.lr, betas=c.betas, eps=1e-8, **fused)


def lr_mult(u, c):
    warm = int(round(c.warmup_frac * c.updates))
    if u < warm:
        return (u + 1) / warm
    p = (u - warm) / max(1, c.updates - warm)
    return c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def flops_per_token_analytic(c):
    """Training FLOPs per token: 3 x forward. Forward = 2 x matmul weights (input embedding excluded) + attention 2 x 2 x T x d per layer."""
    per_layer = 2 * (3 * c.d * c.d + c.d * c.d + 2 * c.d * c.hidden + c.hidden * c.d) + 2 * 2 * c.ctx * c.d
    return 3 * (c.layers * per_layer + 2 * c.d * c.vocab)


def flops_per_token_counter(model, c, device, rows=1):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randint(0, c.vocab, (rows, c.ctx), device=device)
    with FlopCounterMode(display=False) as fc:
        loss = F.cross_entropy(model(x).float().reshape(-1, c.vocab), x.reshape(-1))
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops() / (rows * c.ctx)


# ---------------------------------------------------------------------------------------------------------------------------- data

def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


class Data:
    """Token files from prepare_data.py: train.bin, val.bin (uint16), token_bytes.npy (UTF-8 bytes per token id), meta.json (hashes)."""
    def __init__(self, data_dir, c, check_hashes=True):
        d = Path(data_dir)
        self.meta = json.loads((d / 'meta.json').read_text())
        self.hashes = {}
        if check_hashes:
            for name in ('train.bin', 'val.bin', 'token_bytes.npy'):
                h = sha256_file(d / name); self.hashes[name] = h
                assert h == self.meta['sha256'][name], f'{name}: sha256 {h} does not match meta.json'
        assert self.meta['vocab'] == c.vocab, f'vocab {self.meta["vocab"]} != config {c.vocab}'
        self.train = np.memmap(d / 'train.bin', dtype=np.uint16, mode='r')
        self.val = np.memmap(d / 'val.bin', dtype=np.uint16, mode='r')
        self.token_bytes = np.load(d / 'token_bytes.npy').astype(np.int64)
        self.c = c
        n_windows = (len(self.train) - 1) // c.ctx                 # non-overlapping windows; every training target is used at most once
        need = c.updates * c.rows
        assert need <= n_windows, f'budget needs {need} windows of {c.ctx}, train has {n_windows}: no second epoch allowed'
        self.order = np.random.default_rng(c.seed).permutation(n_windows)[:need]   # the seed sets the data order (same for both arms)
        self.ar = np.arange(c.ctx + 1)

    def batch(self, u):
        starts = self.order[u * self.c.rows:(u + 1) * self.c.rows].astype(np.int64) * self.c.ctx
        w = torch.from_numpy(self.train[starts[:, None] + self.ar].astype(np.int64))
        return w[:, :-1], w[:, 1:]

    def eval_windows(self, split):
        """sealed = the first eval_tokens targets of val; tune = the last eval_tokens targets. They never overlap."""
        W = self.c.ctx + 1; n = math.ceil(self.c.eval_tokens / self.c.ctx)
        assert 2 * n * W <= len(self.val), 'validation text too short for disjoint sealed and tuning windows'
        base = 0 if split == 'sealed' else len(self.val) - n * W
        return [base + i * W for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------------------- run

def autocast(c, device):
    if c.amp == 'fp32':
        return torch.autocast(device.type, enabled=False)
    return torch.autocast(device.type, dtype=torch.float16 if c.amp == 'fp16' else torch.bfloat16)


@torch.no_grad()
def evaluate(model, data, c, device, split):
    model.eval()
    tot_bits, n_tok, n_bytes = 0.0, 0, 0
    starts = data.eval_windows(split)
    for i in range(0, len(starts), c.micro_rows):
        s = np.asarray(starts[i:i + c.micro_rows], dtype=np.int64)
        w = torch.from_numpy(data.val[s[:, None] + data.ar].astype(np.int64))
        x, y = w[:, :-1].to(device), w[:, 1:].to(device)
        with autocast(c, device):
            logits = model(x)
        tot_bits += float(F.cross_entropy(logits.float().reshape(-1, c.vocab), y.reshape(-1), reduction='sum')) / math.log(2)
        n_tok += y.numel(); n_bytes += int(data.token_bytes[w[:, 1:].numpy()].sum())
    model.train()
    return dict(bits_per_token=tot_bits / n_tok, bits_per_byte=tot_bits / n_bytes, eval_targets=n_tok)


def causality(model, c, device, T=None, tol=1e-5, amp=False):
    """Change one token in row 0 of a 2-row batch: no logit of row 0 before that position, and no logit of row 1, may change."""
    T = T or c.ctx
    model.eval()
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, c.vocab, (2, T), generator=g)
    y = x.clone(); t = T // 2; y[0, t] = (y[0, t] + 1) % c.vocab
    with torch.no_grad(), (autocast(c, device) if amp else torch.autocast(device.type, enabled=False)):
        a, b = model(x.to(device)).float(), model(y.to(device)).float()
    model.train()
    prefix, other, at_t = float((a[0, :t] - b[0, :t]).abs().max()), float((a[1] - b[1]).abs().max()), float((a[0, t:] - b[0, t:]).abs().max())
    return dict(ok=prefix <= tol and other <= tol and at_t > 0, max_change_before_t=prefix, max_change_other_row=other,
                change_from_t=at_t, T=T, amp=amp)


def env_info(device):
    e = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
             code_sha256=CODE_SHA, kaggle=os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
    if device.type == 'cuda':
        e.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                 cudnn=torch.backends.cudnn.version(), visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
        try:
            e['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'],
                                             capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as ex:                                                   # noqa: BLE001
            e['nvidia_smi'] = f'unavailable: {ex}'
    return e


def save_json(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n', encoding='utf-8', newline='\n'); os.replace(tmp, path)


def save_logs(out, rec):
    """log.json (everything, including the training-loss curve) and run.jsonl, the raw log that verify.py reads (standard S-3, level 1):
    one 'run' line, one 'eval' line per evaluation point, one 'end' line. Keys sorted, so equal runs give equal bytes."""
    save_json(out / 'log.json', rec)
    c = rec['config']
    head = dict(type='run', check='ECC-004', claim='C12', arm=c['arm'], seed=c['seed'], ctx=c['ctx'], tokens_budget=c['tokens'], config=c,
                params=rec['params'], updates=rec['updates'], flops_per_token=rec['flops_per_token'], code_sha256=CODE_SHA,
                data_sha256=rec['data']['meta'].get('sha256'))
    end = dict(type='end', status=rec['status'], checks=rec['checks'], skipped_steps=rec['skipped_steps'], nonfinite_losses=rec['nonfinite_losses'],
               train_time_s=round(rec['train_time_s'], 3), wall_s=rec.get('wall_s'), segments=rec['segments'])
    lines = [head] + [dict(type='eval', **p) for p in rec['points']] + [end]
    tmp = out / 'run.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n' for x in lines), encoding='utf-8', newline='\n')
    os.replace(tmp, out / 'run.jsonl')


def train(c, data_dir, out, resume=None, deadline_min=None, ckpt_every_min=30.0, device=None, check_hashes=True, log=print, stop_at_update=None):
    t_start = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = Data(data_dir, c, check_hashes=check_hashes)
    torch.manual_seed(c.seed)
    model = LM(c).to(device)
    opt = make_optimizer(model, c, device)
    scaler = torch.amp.GradScaler(device.type, enabled=(c.amp == 'fp16'))
    rec = dict(config=dataclasses.asdict(c), params=n_params(model), updates=c.updates, data=dict(meta=data.meta, sha256_checked=data.hashes),
               flops_per_token=dict(analytic=flops_per_token_analytic(c)), segments=[], points=[], curve=[], checks={},
               skipped_steps=0, nonfinite_losses=0, train_time_s=0.0, status='running')
    u0 = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        assert ck['config'] == dataclasses.asdict(c), 'resume: config differs from the checkpoint'
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        rec = ck['rec']; u0 = ck['update']; rec['status'] = 'running'
        log(f'resumed at update {u0} of {c.updates}')
    else:
        rec['flops_per_token']['flop_counter'] = flops_per_token_counter(model, c, device)
        rec['checks']['causal_init'] = causality(model, c, device)
        rec['checks']['causal_init_amp'] = causality(model, c, device, amp=True, tol=float('inf'))
    rec['segments'].append(dict(start_update=u0, env=env_info(device), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    eval_at = sorted({max(1, int(round(f * c.updates))) for f in c.eval_fracs} | {c.updates})
    n_micro = c.rows // c.micro_rows; assert n_micro * c.micro_rows == c.rows
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    last_ckpt, run_loss, run_n, consecutive_bad = time.time(), 0.0, 0, 0

    def loss_fn(x, y):
        with autocast(c, device):
            logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, c.vocab), y.reshape(-1))
    # Compilation time is part of training time (it happens in the first updates). Evaluation and the causality test use the eager model.
    step_loss = loss_fn if c.compile == 'none' else torch.compile(loss_fn, backend=c.compile)

    def checkpoint(u):
        tmp = out / 'ckpt.pt.tmp'
        torch.save(dict(config=dataclasses.asdict(c), model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        update=u, rec=rec), tmp)
        os.replace(tmp, out / 'ckpt.pt'); save_logs(out, rec)

    def eval_point(u, kind):
        ev = evaluate(model, data, c, device, c.eval_split)
        rec['points'].append(dict(update=u, kind=kind, tokens=u * c.tokens_per_update, flops=u * c.tokens_per_update * rec['flops_per_token']['analytic'],
                                  train_time_s=round(rec['train_time_s'], 3), **{k: round(v, 6) if isinstance(v, float) else v for k, v in ev.items()}))
        log(f'  eval u {u}: {ev["bits_per_token"]:.4f} bits/token, {ev["bits_per_byte"]:.4f} bits/byte, train time {rec["train_time_s"]:.0f} s')
        save_logs(out, rec)

    for u in range(u0, c.updates):
        if deadline_min and (time.time() - t_start) / 60 > deadline_min:
            eval_point(u, 'stop'); rec['status'] = f'incomplete: deadline at update {u}'; checkpoint(u); return rec
        if stop_at_update is not None and u >= stop_at_update:                       # planned segment end (tests, session planning)
            rec['status'] = f'incomplete: segment ended at update {u}'; checkpoint(u); return rec
        xs, ys = data.batch(u)
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        m = lr_mult(u, c)
        for gr in opt.param_groups:
            gr['lr'] = c.lr * m
        sync(); t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for i in range(n_micro):
            x, y = xs[i * c.micro_rows:(i + 1) * c.micro_rows], ys[i * c.micro_rows:(i + 1) * c.micro_rows]
            loss = step_loss(x, y) / n_micro
            scaler.scale(loss).backward()
            loss_sum += loss.detach()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if c.arm == 'ngpt':
            model.renormalize()
        sync(); dt = time.perf_counter() - t0; rec['train_time_s'] += dt
        if u == u0:
            rec['segments'][-1]['first_update_s'] = round(dt, 3)                    # includes compilation when compile != none
        lv = float(loss_sum)
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            rec['skipped_steps'] += 1
        if not math.isfinite(lv):
            rec['nonfinite_losses'] += 1; consecutive_bad += 1
            if consecutive_bad >= 50:
                rec['status'] = f'rejected: 50 consecutive non-finite losses at update {u + 1}'; checkpoint(u + 1); return rec
        else:
            consecutive_bad = 0; run_loss += lv; run_n += 1
        if (u + 1) % c.log_every == 0:
            rec['curve'].append(dict(update=u + 1, loss=round(run_loss / max(1, run_n), 5), lr=c.lr * m, grad_norm=round(float(gnorm), 4),
                                     scale=scaler.get_scale() if scaler.is_enabled() else None, train_time_s=round(rec['train_time_s'], 3)))
            run_loss, run_n = 0.0, 0
        if u + 1 in eval_at:
            eval_point(u + 1, 'scheduled')
        if (time.time() - last_ckpt) / 60 > ckpt_every_min:
            checkpoint(u + 1); last_ckpt = time.time()
    rec['checks']['causal_trained'] = causality(model, c, device)
    rec['checks']['finite'] = dict(ok=rec['nonfinite_losses'] == 0, nonfinite_losses=rec['nonfinite_losses'], skipped_steps=rec['skipped_steps'])
    rec['status'] = 'done'; rec['wall_s'] = round(time.time() - t_start, 1)
    checkpoint(c.updates)
    return rec


def timing_block(cfgs, data_dir, out, n_up=60, skip=10, device=None, log=print):
    """Per-update GPU time of each arm ALONE on one GPU, in the order A B B A (drift cancels), after the long runs of a session.
    R_time (ESTIMATED) = R_steps x (nGPT seconds per update / baseline seconds per update); the long runs themselves may share the
    machine with another run and are only reported. cfgs: {'baseline': Config, 'ngpt': Config} with the session's real settings."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(out); out.mkdir(parents=True, exist_ok=True); lines = []
    for i, arm in enumerate(('baseline', 'ngpt', 'ngpt', 'baseline')):
        c = dataclasses.replace(cfgs[arm], tokens=float(n_up * cfgs[arm].tokens_per_update), log_every=1, eval_fracs=(1.0,), eval_tokens=2 ** 14,
                                eval_split='tune')
        rec = train(c, data_dir, out / f'timing_{i}_{arm}', device=device, check_hashes=False, log=lambda s: None)
        cur = rec['curve']; t = sorted([b['train_time_s'] - a['train_time_s'] for a, b in zip(cur, cur[1:])][skip:])   # drop warm-up and compilation
        lines.append(dict(type='timing', order=i, arm=arm, seed=c.seed, ctx=c.ctx, d=c.d, layers=c.layers, micro_rows=c.micro_rows, compile=c.compile, n=len(t),
                          median_s=round(t[len(t) // 2], 5), mean_s=round(sum(t) / len(t), 5), gpu=rec['segments'][0]['env'].get('gpu'),
                          code_sha256=CODE_SHA))
        (out / f'timing_{i}_{arm}' / 'ckpt.pt').unlink(missing_ok=True)
        log(f'timing {i} {arm}: median {lines[-1]["median_s"]} s per update')
    (out / 'timing.jsonl').write_text(''.join(json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n' for x in lines), encoding='utf-8', newline='\n')
    return lines


def config_from_args(a):
    kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(Config) if getattr(a, f.name, None) is not None}
    return Config(**kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'checks', 'flops'])
    for f in dataclasses.fields(Config):
        if f.type in (int, 'int'):
            p.add_argument('--' + f.name.replace('_', '-'), dest=f.name, type=int)
        elif f.type in (float, 'float'):
            p.add_argument('--' + f.name.replace('_', '-'), dest=f.name, type=float)
        elif f.type in (str, 'str'):
            p.add_argument('--' + f.name.replace('_', '-'), dest=f.name, type=str)
    p.add_argument('--data-dir'); p.add_argument('--out'); p.add_argument('--resume'); p.add_argument('--deadline-min', type=float)
    a = p.parse_args(); c = config_from_args(a)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.cmd == 'train':
        rec = train(c, a.data_dir, a.out, resume=a.resume, deadline_min=a.deadline_min)
        print(json.dumps(dict(status=rec['status'], final=rec['points'][-1] if rec['points'] else None, train_time_s=rec['train_time_s'])))
    else:
        torch.manual_seed(c.seed); model = LM(c).to(device)
        if a.cmd == 'checks':
            print(json.dumps(dict(params=n_params(model), causal=causality(model, c, device), causal_amp=causality(model, c, device, amp=True,
                                                                                                                  tol=float('inf')))))
        else:
            print(json.dumps(dict(params=n_params(model), analytic=flops_per_token_analytic(c), flop_counter=flops_per_token_counter(model, c, device))))


if __name__ == '__main__':
    main()
