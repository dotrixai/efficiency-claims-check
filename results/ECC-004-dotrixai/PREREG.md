# Pre-registration: C12 nGPT on GPU at 25M and 89M parameters, context 1k and 4k, a small-scale screen (ECC-004)

**DRAFT, not frozen.** Frozen at: (the commit time of this file on main, once review is done)
Author: DotrixAI (code and runs by a DotrixAI agent session, owner Noven). Reviewer: to be named.

**Scale, up front:** this is a proof of concept, a small-scale screen at about 25M parameters, with one point at 89M to show the trend. It is not a verdict on nGPT at the paper's 0.5B to 1B. Every verdict below carries that label.

**In short:**
- The claim: nGPT needs 4x fewer training steps than a normal GPT at context 1k, and 10x fewer at 4k.
- The test: train both on the same OpenWebText data, on free Kaggle T4s, and count how many steps nGPT needs to reach the baseline's final loss. 25M parameters at 1k and 4k with 3 seeds each, and one 89M run at 1k.
- Fairness: the same shapes and data on both sides, a modern baseline with QK-norm, the same learning-rate search on both sides, and time measured with each model alone on the GPU.
- Everything is written down before the runs, and anyone can recompute the result from the logs with `verify.py`.
- Whatever comes out gets published, including "it did not hold".

## Why this check

ECC-001 tested the same claim on a laptop CPU with a 0.73M model, context 128 and 1,200 updates. It came out PARTIAL (1.47x fewer steps against the claimed 4x to 20x). Two fair objections followed. A GPU method run on a CPU says little about how it behaves where it is meant to run. And 0.73M at context 128 is far from the setting where the paper reports its gains, which grow with context. This check redoes it on free Kaggle GPUs, about 35x bigger, at context 1024 and 4096, with the baseline trained much closer to convergence. It is still small; the next section says what that means.

## Why this may or may not transfer

| | paper | this check |
|---|---|---|
| model | 0.5B and 1B | about 25M, plus one 89M point |
| data | OpenWebText | OpenWebText (12 of 80 files) |
| tokens per run | up to about 400B | 504M at 25M (20 per parameter); 446M at 89M (5 per parameter) |
| context | 1k, 4k, 8k | 1k, 4k |
| vocabulary | 32k (LLaMA-2) | 8k |
| hardware, precision | 64 A100s, bf16 weights | 2 T4s, fp16 compute with fp32 weights |
| baseline | GPT without QK-norm | GPT with QK-norm |

Reasons a result here may hold at larger scale:
- the mechanism the paper credits (unit-norm weights and hidden states, learned per-dimension step sizes) does not depend on width in an obvious way;
- the claimed gain is tied to context length, and both claimed contexts are tested here.

Reasons it may not:
- efficiency gains often change with scale; Muon's gain over a tuned AdamW shrinks as models grow (C01, C02), and the nGPT authors themselves report about 2x at 30B against 4x to 20x at 1B;
- a small model trained for 504M tokens sits in a different regime from a 1B model trained for hundreds of billions;
- the vocabulary, the precision and the baseline all differ from the paper.

What a 25M result cannot show: whether nGPT saves steps at 0.5B to 1B or beyond, whether a gain grows or shrinks with size, or anything at 8k. A 1B run at 20 tokens per parameter needs about 1.2e20 FLOPs (ESTIMATED), roughly 3,000 T4 hours, which free GPUs cannot reach. That is why this is a screen, and why it adds a second size: a trend over two sizes says more than one point, though two points are still only a hint of a trend.

## Claim

- **C12:** "The normalized Transformer (nGPT) needs 4x to 20x fewer training steps."
- **Source:** Loshchilov, Hsieh, Sun, Ginsburg, *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*, [arXiv 2410.01131](https://arxiv.org/abs/2410.01131) (2024). The paper reports about 4x, 10x and 20x fewer tokens to the same validation loss at context 1k, 4k and 8k (its Figure 2), for 0.5B and 1B models on OpenWebText.
- **Tested here:** 4x at 1k (R = 0.25) and 10x at 4k (R = 0.10). 8k is not tested.
- **Context:** the authors' later paper *Training nGPT* ([arXiv 2608.01284](https://arxiv.org/abs/2608.01284), 2026) reports about 2x fewer tokens for a 30B hybrid MoE model against AdamW. NVIDIA's public reimplementation ([github.com/NVIDIA/ngpt](https://github.com/NVIDIA/ngpt)) says its own speed-ups are larger than they should be, because bf16 parameters hurt the GPT baseline more than nGPT. Here both sides keep fp32 parameters.

## Setup

### Model (same shapes on both sides)

**Main size, 25M:** d 384, 8 layers, 6 heads (head dim 64), SwiGLU MLP with hidden size 1536 (4d, as in the paper), RoPE, untied input and output embeddings, vocabulary 8,192. Parameters: baseline 25,179,904, nGPT 25,207,808. The difference is nGPT's scaling vectors (41,984) against the baseline's LayerNorm and QK-norm gains (14,080). FLOPs per token are the same on both sides: 169.9M at context 1024 and 283.1M at 4096 (forward and backward, analytic count, attention counted at full T x T).

**Scale point, 89M (context 1024 only):** d 640, 12 layers, 10 heads (head dim 64), MLP hidden 2560, otherwise the same. About 89.1M parameters and 597.6M FLOPs per token at 1024 (3.5x the 25M model).

### Baseline: modern GPT

Pre-LayerNorm blocks, RoPE, SwiGLU, untied embeddings, and **QK-norm** (per-head RMSNorm of q and k before RoPE, with a learnable gain). AdamW (betas 0.9 and 0.95, eps 1e-8), weight decay 0.1 on matrices except the input embedding, linear warmup over the first 1% of updates, cosine decay to 0, gradient clipping at 1.0. Init: normal std 0.02, residual projections scaled by 1/sqrt(2 x layers).

This is ECC baseline v1 scaled up, plus QK-norm. QK-norm is now common in small modern recipes, and without it a reader could fairly ask whether nGPT only wins because it normalizes q and k. The paper's own GPT baseline has no QK-norm, so this baseline is a little stronger than the paper's. This choice is argued here as METHOD §3 asks, and is open for review.

### Method: nGPT

From the paper's equations, checked line by line against NVIDIA's reimplementation (commit ed19eb2):

- every hidden state on the unit sphere: h ← Norm(Norm(h) + |α| (Norm(h_X) − Norm(h))), after attention and after the MLP;
- q and k normalized after RoPE, times s_qk (one vector per head), softmax scale sqrt(d_head);
- SwiGLU input times s_uv · sqrt(d);
- logits times s_z;
- no LayerNorm, no weight decay, no warmup, cosine decay to 0, gradient clipping at 1.0, Adam betas 0.9 and 0.95;
- all matrices renormalized along the embedding dimension at init and after every optimizer step;
- scaling vectors with the paper's init and scale: α (0.05, 1/sqrt(d)), s_qk (1, 1/sqrt(d)), s_uv (1, 1), s_z (1, 1/sqrt(d)).

Known differences from the paper:
- vocabulary 8,192, against LLaMA-2's 32k (at 25M parameters a 32k vocabulary would make the embedding and head about half the model);
- about 25M parameters, against 0.5B and 1B;
- fp16 compute with fp32 parameters on a T4, against bf16 on A100s;
- token budget 504M, against tens to hundreds of billions;
- the scaling of u by sqrt(d) follows the code; it has no effect, since the MLP output is normalized next;
- the base scale of the scaling vectors is 1/sqrt(d) as the paper says (1/sqrt(384) at 25M). NVIDIA's code sets it by default to 1/sqrt(1024) (a config value), which equals 1/sqrt(d) only for its d 1024 models. The choice changes how fast the scaling vectors learn;
- α starts at 0.05, as in the paper's main text and the code; the paper's appendix gives 1/sqrt(d) instead.

Code: `ecc004.py` (both sides).

### Data and tokenizer

- OpenWebText, Hugging Face `Skylion007/openwebtext`, revision 79d93d786212f7344586290adb811d4ae6a1762c (CC0). Every file is checked against its SHA-256.
- Split by file, before any text was read:
  - training text: files 00000 to 00011, 1,202,073 documents, 1,593,562,875 tokens;
  - validation text: file 00079, the first 12,000 documents that are not exact copies of a training document (none were), 15,762,071 tokens.
- Byte-level BPE with 8,192 tokens, trained on the first 300,000 training documents only. Each document is followed by a separator token. 3.75 bytes per token.
- Token files (SHA-256): train.bin `7bec0adad02166721cd5d120ef606404ad1abc0de88b63a27f0768acaa14f632`, val.bin `e9ae679cb3886c25b6b29bfb233e4e9901354b32e931666acb0bbd2fc1bd3368`, token_bytes.npy `48236aca118c9dd4a550c2255df92e040d593779c0b0f2f43e4d168493328373`, tokenizer.json `b2c0731eef7b4c53ba274fe274818ee2f9033a45f3d1a25c234da57e1f207cd3`.
- Rebuilt by `prepare_data.py` (a Kaggle CPU kernel, 29 minutes).

### Training

- 2^17 = 131,072 tokens per update: 128 rows of 1024, or 32 rows of 4096.
- Full budget at 25M: 3,842 updates = 503,578,624 tokens, about 20 tokens per parameter.
- Full budget at 89M: 3,400 updates = 445,644,800 tokens, about 5 tokens per parameter. 20 tokens per parameter would need 1.78B tokens, more than the 1.59B in the training text, and would break the one-epoch rule.
- **Matching point at 25M:** a 25M baseline at 1/4 of its budget (960 updates, also about 5 tokens per parameter), so the two sizes are compared at the same tokens per parameter. Training windows are non-overlapping and never repeat (one epoch at most). The seed sets the window order and the init; both sides with the same seed see the same data in the same order.
- fp16 autocast with fp32 parameters and optimizer state, GradScaler. Norms, residual mixing and the loss run in fp32.
- torch.compile (inductor) of the forward pass and loss, for both sides. Compilation time counts as training time.

### Evaluation

- **Sealed windows:** the first 2,097,152 target tokens of the validation text, in non-overlapping windows of the training context. Every target is scored. Used for Q and the verdict only.
- **Tuning windows:** the last 2,097,152 target tokens of the validation text. Used for tuning, calibration and pilots only.
- Evaluation points at 1, 2, 3, 5, 7.5, 10, 15, 20, then every 5% up to 100% of each run's own budget.
- Reported in bits per token, and in bits per byte.

### Tuning (same effort on both sides)

- Done at 1/4 of the full budget (960 updates), seed 7, scored on the tuning windows at the end of the run.
- **Context 1024:** 4 learning rates per side, each a doubling of the last.
  - Baseline: {1e-3, 2e-3, 4e-3, 8e-3}.
  - nGPT: {4e-3, 8e-3, 1.6e-2, 3.2e-2}.
  - The grids sit in different places because the calibration (below) showed nGPT's best rate is several times the baseline's. The number of points and the rules are the same.
- **Context 4096 (25M):** 3 rates per side: half, equal to and double the side's best rate at 1024.
- **89M (context 1024):** 3 rates per side at 1/4 of its budget: a quarter, half and equal to the side's best rate at 25M (larger models usually want lower rates).
- **Edge rule:** if the best rate is at an edge of its grid, one more doubling is run on that side. This is allowed at most twice per side and context, and reported.
- A run that diverges is dropped from the grid and reported, not retried.
- **Compile guard:** at 1024, one baseline run without torch.compile at the chosen rate, same budget and seed. If it ends more than 0.10 bits per token below the compiled run, both sides switch to eager mode for all final runs, and the result says so.
- Only the learning rate is tuned, on both sides. Warmup, weight decay, Adam betas and the schedule are fixed as written above.
- Tuning at 1/4 of the budget is a limit: the best rate for the full budget may differ a little.

### Final runs

- **25M:** seeds 11, 22 and 33 at context 1024 (stage A) and 4096 (stage B). **89M:** seed 11 at context 1024 (stage C).
- Per seed and stage, one Kaggle session with two T4s:
  - the baseline at the full budget (at 25M / 1024 also at 1/4, the matching point);
  - nGPT at the full budget and at 1/2, 1/4 and 1/8 of it, each with its own full schedule (at 25M / 4096 also 1/16; at 25M / 1024 also 1/16 and 1/32, which bracket the matching point).
- The runs are split over the two GPUs by `session.py`. The GPU assignment is swapped between seeds.
- Then a **timing block** on one GPU: each side alone for 60 updates, in the order baseline, nGPT, nGPT, baseline. The first 10 updates are dropped (warm-up and compilation), and the median seconds per update is taken.

### Hardware and versions

Kaggle "GPU T4 x2". Calibration ran on Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04. Every run logs the GPU name, driver, CUDA and torch versions. A run on different versions is reported as a deviation.

### Pilot runs before freezing (all scored on the tuning windows; none used for a verdict)

1. **Calibration 1** (2026-10-20, 1.23 GPU hours), baseline without QK-norm, eager mode:
   - causality passed on the GPU for both sides at 1024 and 4096;
   - seconds per update: baseline 2.21 (1k) and 3.29 (4k), nGPT 4.80 and 5.78;
   - short fp16 runs: nGPT was unstable at 6.4e-2, stable at 1.6e-2; the baseline was stable at 2e-3 and 8e-3.
2. **Calibration 2** (2026-10-20, 0.50 GPU hours), torch.compile:
   - seconds per update: baseline 1.40 (1k) and 2.61 (4k), nGPT 1.83 and 2.89;
   - two runs side by side on the two GPUs slowed the baseline by 13% to 21% and nGPT by up to 9%; this is why R_time comes from the timing block;
   - compiled and eager nGPT agreed over 60 updates; the compiled baseline was 0.23 nats worse at that point.
3. **Compile diagnostic** (2026-10-21, 0.29 GPU hours):
   - one update from identical weights and batch: compiled equal to eager within noise, on both sides, in fp32 and fp16;
   - over 200 updates, an eager baseline from calibration 1 lies between two compiled runs that differ only in micro-batch split (difference within 0.03 to 0.05 nats);
   - the 60-update gap is not fully explained; the compile guard above covers it.

4. **QK-norm baseline pilot** (2026-10-21, 0.39 GPU hours), compiled:
   - causality passed at 1024 and 4096;
   - one update compiled vs eager: equal within noise;
   - 200 updates at 1024 (tuning windows, bits per token): 8.64 at 2e-3, 9.18 at 8e-3; 3.2e-2 diverged at about update 58. The baseline grid was therefore kept;
   - about 1.5 seconds per update (ESTIMATED from these runs, compilation included), about 10% slower than without QK-norm.
5. **89M calibration** (2026-10-21, 0.39 GPU hours; a first try was cancelled after it hung with no output, and Kaggle's quota page suggests it never got a GPU):
   - causality passed for both sides at 1024; 89,162,496 and 89,221,632 parameters;
   - compiled seconds per update: baseline 5.18 (micro-batch 16), nGPT 5.84 (micro-batch 8), so nGPT is 1.13x slower per step; peak memory at most 9.6 GB.

The data order of seed 11 was used in the calibration runs (first 200 updates), scored on the tuning windows only.

## Target quality

- **Q** = the baseline's bits per token on the sealed windows at its last evaluation point, per seed, context and size (and for the 25M matching point, at the end of the 1/4-budget baseline).

## What is measured

- **R_steps** = training tokens nGPT needs to reach Q, divided by the baseline's. Tokens per update are equal, so this is the ratio of steps the paper reports. FLOPs per token are equal too, so **R_FLOP = R_steps**.
- **Primary estimator, as in the paper:** the final points of the nGPT runs at their different budgets, each with its own schedule, interpolated linearly in log tokens.
- **Trend with size:** R at 89M divided by R at 25M, both at 5 tokens per parameter (seed 11, and the 25M seeds' range). Reported with its seed count; no pass rule.
- **Secondary estimator, as in METHOD §1 and ECC-001:** the evaluation points of the full-budget nGPT run. Reported, not used for the verdict.
- A quality never reached gives a lower bound. A quality already passed at the smallest budget gives an upper bound.
- **R_time (ESTIMATED)** = R_steps x (nGPT seconds per update / baseline seconds per update) from the timing block. The training time measured inside the long runs is reported too, but runs on the two GPUs can slow each other, so it is not used for the verdict.
- Everything is recomputed from the logs by `verify.py`, which writes `result.json`.

## Prediction (written before any final run)

| | 25M, 1024, 20 tok/param | 25M, 4096, 20 tok/param | 25M, 1024, 5 tok/param | 89M, 1024, 5 tok/param |
|---|---|---|---|---|
| median R (primary estimator), 80% interval | 0.6, [0.35, 0.95] | 0.55, [0.25, 1.0] | 0.5, [0.25, 0.85] | 0.55, [0.25, 0.95] |
| median R_time | 0.75, [0.45, 1.2] | 0.6, [0.3, 1.1] | 0.6, [0.3, 1.05] | 0.65, [0.3, 1.1] |
| HOLDS / PARTIAL / DOES NOT HOLD / INCONCLUSIVE | 0.10 / 0.60 / 0.20 / 0.10 | 0.05 / 0.55 / 0.25 / 0.15 | 0.15 / 0.60 / 0.15 / 0.10 | 0.10 / 0.55 / 0.20 / 0.15 |

Trend: R at 89M over R at 25M (both at 5 tokens per parameter), median guess 1.1, 80% interval [0.7, 1.6]. Above 1 means the gain shrinks with size.

Reasoning: ECC-001 measured 1.47x fewer steps at small scale, with a large early gain that faded (so a shorter budget, 5 tokens per parameter, should look better for nGPT); the authors' own later number at large scale is about 2x; NVIDIA's reimplementation notes its speed-ups are inflated by bf16; and our baseline has QK-norm.

## Pass rule

Per context, using the median over seeds of R (primary estimator) and R_time:

| context | claim | HOLDS | PARTIAL | DOES NOT HOLD |
|---|---|---|---|---|
| 1024 | 4x (R 0.25) | R ≤ 0.25 and R_time ≤ 0.35 | R in (0.25, 0.8], or R ≤ 0.25 with R_time > 0.35 | R > 0.8 |
| 4096 | 10x (R 0.10) | R ≤ 0.10 and R_time ≤ 0.14 | R in (0.10, 0.8], or R ≤ 0.10 with R_time > 0.14 | R > 0.8 |

**INCONCLUSIVE** if a validity check fails, if the seeds fall in different R classes, or if a bound leaves the class open.

A context with fewer than 3 finished seeds gets its verdict with the seed count, for example "PARTIAL (1 seed only)".

The pass rule of a context applies to every stage at that context (25M at 20 and at 5 tokens per parameter, 89M at 5). Every verdict is stated with its scale, budget and the screen label, for example "PARTIAL at 25M, context 1024, 20 tokens per parameter (small-scale screen)"; the 89M stage always carries "(1 seed only)". It is a verdict on this setup, not on the claim at 0.5B to 1B.

## Validity checks

A run is invalid if any of these fail:
- **causality:** change one token in one row; nothing earlier in that row and nothing in the other row may change (fp32, 1e-5), on the initial and the trained model;
- **finite training:** no non-finite loss; fp16 steps that the GradScaler skips (gradient overflow) are allowed, counted and reported;
- **data:** token-file hashes match, no second epoch, sealed windows never scored before the final runs;
- **versions:** code hashes logged in every run.

## Stop rules and budget

- **Approved GPU budget:** 48 Kaggle GPU hours for tuning and final runs (owner, 2026-10-21), on DotrixAI's Kaggle account. The calibration and pilot hours above are outside it.
- **Order of work** (planned hours, PROJECTED):
  1. 25M tuning at 1024, with edge runs and the compile guard (about 3);
  2. the three 25M seeds at 1024, with the matching point (about 11);
  3. 25M tuning at 4096 (about 2.5);
  4. 25M seed 11 at 4096 (about 5.5);
  5. 89M tuning (about 5);
  6. 89M seed 11 (about 10.5);
  7. 25M seeds 22 and 33 at 4096 (about 11).
  This order keeps at least one 4096 seed and the 89M point if the budget runs short.
- **If the budget runs out:** the seeds not yet run are reported as not run, never dropped silently. If no 4096 seed finishes, stage B is reported as not run; the same for the 89M point.
- **Session limit:** a run still going 11 hours into its Kaggle session stops with an evaluation at that point. It is resumed from its checkpoint in a new session if the budget allows; otherwise that point is reported as a bound.
- **Nothing else stops a run.** Quality is never used to stop a run.
- **After the freeze:** no change to data, code, grids, seeds or rules. Corrections go into RESULT.md and DEVIATIONS.md.

## Reproducibility (standard S-3)

- **Verifiable:**
  - raw logs per run (`run.jsonl`, one line per evaluation point) and per timing block (`timing.jsonl`), with their SHA-256 in RESULT;
  - `python verify.py --check` recomputes every ratio and the verdict on a CPU in seconds.
- **Re-runnable:**
  - a Kaggle notebook runs one seed at one context (`REPRODUCE.md`);
  - the token files are rebuilt deterministically by `prepare_data.py`;
  - the tolerance for a rerun is set from the spread of the three seeds: within the range of the seeds' final bits per token plus 0.02, and within the range of their R plus 0.05;
  - GPU hours per seed are reported.
- **Buildable:** `result.json` and a draft technique note.

## Limits known before the run

- Two model sizes (about 25M, and one 89M seed at context 1024 only) and one data set. The paper's gains are reported at 0.5B to 1B.
- The 89M point is trained for only 5 tokens per parameter, because the data has no room for 20 without a second epoch.
- A budget of 20 tokens per parameter, far below the paper's.
- T4 GPUs in fp16.
- An 8k vocabulary.
- Only the learning rate is tuned, at 1/4 of the budget, with grid positions informed by calibration.
- The baseline has QK-norm, which the paper's baseline does not.

## Separation

This check uses public data and published code only. It is independent of DotrixAI's confidential research: no internal results, values or predictions from it are used here.
