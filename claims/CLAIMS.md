# Claims to check

Generated from `claims.csv` by `tools/make_claims_md.py`. Do not edit by hand.

Each row is a published claim as we read it from the source. Check the source before relying on a row.
"Claimed ratio" is compute of the method divided by compute of the baseline at equal quality, so below 1 means cheaper.

Status: **OPEN** = no independent matched compute check found yet; **SUPPORTED** = independent work points the same way; **CONTESTED** = independent work disagrees; **SHRINKS_WITH_SCALE** = holds, but the gain shrinks at larger scale; **CHECKED** = checked in this project; the verdict and its scale are in the row and in results/.

## Optimizer

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C01 | Muon trains NanoGPT about 1.35x faster than AdamW; a 1.5B model in 10 instead of 13.3 hours. | [Jordan, Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) (2024) | 0.74 | 124M to 1.5B | Wen et al. 2025 (C02) find matrix optimizer speedups over a well tuned AdamW shrink with scale. | SHRINKS_WITH_SCALE | yes |
| C02 | Matrix optimizers beat a well tuned AdamW by about 1.4x at 0.1B, falling to about 1.1x at 1.2B. | [Wen, Hall, Ma, Liang, Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) (2025) | 0.71 to 0.91 | 0.1B to 1.2B | This is itself a controlled comparison of ten optimizers with separate tuning. | SUPPORTED | yes |
| C03 | Muon is about 2x as compute efficient as AdamW for LLM training. | [Liu et al., Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (2025) | 0.5 | Moonlight models | Wen et al. 2025 (C02) measure about 1.1x at 1.2B against a tuned AdamW. | CONTESTED | yes |
| C04 | Sophia reaches the same loss as Adam with about half the steps. | [Liu et al., Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342) (2023) | 0.5 | 125M to 1.5B | Wen et al. 2025 (C02) do not find a 2x gain against a tuned AdamW. | CONTESTED | yes |
| C05 | SOAP needs about 40% fewer iterations and 35% less wall clock time than AdamW at large batch. | [Vyas et al., SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321) (2024) | 0.6 to 0.65 | 360M and 660M | Included in the comparison of Wen et al. 2025 (C02). | SUPPORTED | yes |
| C06 | AdEMAMix trained on 101B tokens matches AdamW trained on 197B tokens. | [Pagliardini, Ablin, Grangier, The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137) (2024) | 0.51 | 1.3B | None found yet. | OPEN | yes |

## Schedule

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C07 | Sequence length warmup cuts tokens by up to 2.2x and wall clock time by up to 3.7x. | [Li, Zhang, He, The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models](https://arxiv.org/abs/2108.06084) (2022) | 0.45 | GPT-2 and GPT-3 style 125M to 1.3B | Sequence length warmup is part of strong small model recipes; gains on top of such recipes are not reported in the source. | SUPPORTED | yes |
| C08 | A constant learning rate with a short cooldown matches a cosine schedule, which makes scaling studies cheaper. | [Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392) (2024) | about 1.0 per run | up to about 1B | Warmup stable decay schedules are widely used in later work. | SUPPORTED | yes |

## Architecture

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C09 | Value residual learning reaches the same loss with about 16% fewer parameters and 20% less data. | [Zhou et al., Value Residual Learning (ResFormer)](https://arxiv.org/abs/2410.17897) (2024) | about 0.8 | see paper | Related value embeddings are used in the modded-nanogpt speedrun; not the same mechanism. | OPEN | yes |
| C10 | MUDDFormer matches a Transformer trained with 1.8x to 2.4x the compute, at about 0.4% extra compute. | [Xiao et al., MUDDFormer: Multiway Dynamic Dense Connections](https://arxiv.org/abs/2502.12170) (2025) | 0.42 to 0.56 | Pythia scale | None found yet. | OPEN | yes |
| C11 | Selective attention gives quality equivalent to about twice the attention heads and parameters. | [Leviathan, Kalman, Matias, Selective Attention Improves Transformer](https://arxiv.org/abs/2410.02703) (2024) | not stated as a compute ratio | C4 | None found yet. | OPEN | yes |
| C12 | The normalized Transformer (nGPT) needs 4x to 20x fewer training steps. | [Loshchilov et al., nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131) (2024) | 0.05 to 0.25 | see paper | We found no adoption in strong small model recipes. ECC-001 (DotrixAI, 2026-10-17): PARTIAL at 0.73M params, context 128, 3 seeds, laptop CPU. R_FLOP 0.68 (1.47x fewer steps; claim 4x to 20x), R_time 0.88; the gain was 4.9x early in training and faded. See results/ECC-001-dotrixai. | CHECKED | yes |
| C13 | Short dynamic convolutions give the quality of 1.33x to 1.6x the compute. | [Sieberling, Runwal, Panda, Kim, Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825) (2026) | 0.63 to 0.75 | see paper | Related static short convolutions (Canon layers) exist; no independent matched compute check found. ECC-002 (Shikhar, exploratory, 2026-10-18): R_FLOP 0.56 against a simple baseline, R_time 0.97; not an official check. See results/ECC-002-shikhar. | OPEN | yes |
| C14 | A sigmoid gate after attention lowers perplexity (7.499 to 7.404 at 1.7B on 400B tokens) and tolerates larger learning rates. | [Qiu et al., Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) (2025) | about 0.87 to 0.9 (our estimate from perplexity) | 1.7B dense and 15B MoE | The source reports no results below 1.7B. | OPEN | yes |
| C15 | Hyena reaches Transformer quality with 20% less training compute at 2K context. | [Poli et al., Hyena Hierarchy](https://arxiv.org/abs/2302.10866) (2023) | 0.8 | baselines of 2023 | Baselines have improved since 2023; not rechecked against a modern recipe. | OPEN | yes |
| C16 | Replacing about 63% of attention blocks with feed forward blocks keeps WikiText-103 perplexity and cuts time by about 35%. | [Mandava, Migacz, Fit Florea, Pay Attention when Required (PAR Transformer)](https://arxiv.org/abs/2009.04534) (2020) | 0.65 (time) | Transformer-XL scale | Later work also removes attention layers after training. | SUPPORTED | yes |
| C24 | Energy-based Transformers scale up to 35% faster than Transformer++. | [Gladstone et al., Energy-Based Transformers are Scalable Learners and Thinkers](https://arxiv.org/abs/2507.02092) (2025) | scaling rate claim | see paper | None found yet. | OPEN | partly |
| C25 | A byte level hierarchical network (H-Net) beats a BPE Transformer at matched compute and data. | [Hwang, Wang, Gu, Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) (2025) | about 0.5 (parameters) | see paper | None found yet. | OPEN | partly |

## Data

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C17 | Pre-pretraining on a formal language (k-Shuffle Dyck) reaches the same loss with 33% fewer natural language tokens. | [Hu, Petty, Shi, Merrill, Linzen, Between Circuits and Chomsky](https://arxiv.org/abs/2502.19249) (2025) | about 0.69 | 1B | None found yet. | OPEN | yes |
| C18 | Pre-pre-training on neural cellular automata data speeds convergence 1.4x to 1.6x. | [Lee, Han, Kumar, Agrawal, Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055) (2026) | 0.63 to 0.72 | 400M to 1.6B | None found yet. | OPEN | yes |

## Objective

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C19 | Multi-token prediction helps more as models grow and hurts small models. | [Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) (2024) | above 1 at small scale | up to 13B | Multi-token prediction is used in later large models. | SUPPORTED | yes |
| C23 | Masked diffusion language models need about 16x the compute to match autoregressive models, but do better when data is scarce. | [Nie et al., Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) (2024) | about 16 | up to 1.1B | A useful negative control: a claimed cost above 1. | SUPPORTED | yes |

## Memory

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C20 | A 400M model with a 12.8M entry n-gram input vocabulary matches a 1B baseline. | [Huang et al., Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](https://arxiv.org/abs/2501.16975) (2025) | about 0.4 (parameters) | OLMo2 151M to 1B | Related hashed n-gram input embeddings appear in small model speedrun recipes. | SUPPORTED | yes |
| C21 | Memory layers match dense models that use 2x to 4x the compute. | [Berges et al., Memory Layers at Scale](https://arxiv.org/abs/2412.09764) (2024) | 0.25 to 0.5 | 1.3B | Memory bandwidth can dominate real cost; time ratios are rarely reported. | OPEN | partly |
| C22 | A 12 layer model with a product key memory beats a 24 layer model without one. | [Lample et al., Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) (2019) | about 0.5 (parameters) | 2019 baselines | Basis of later memory layer work (C21). | SUPPORTED | partly |

## Batch

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C26 | Batch size warmup reaches the same loss with 43% fewer gradient steps. | [Merrill et al., Critical Batch Size Revisited](https://arxiv.org/abs/2505.23971) (2025) | steps, not compute | OLMo 1B and 7B | Steps saved are not compute saved; a good case for careful accounting. | SUPPORTED | partly |

## Search

| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |
|---|---|---|---|---|---|---|---|
| C27 | Successive halving and Hyperband, which give more training budget to promising hyperparameter settings, find good settings an order of magnitude faster than baseline methods such as random search. | [Jamieson and Talwalkar, Non-stochastic Best Arm Identification and Hyperparameter Optimization (arXiv 1502.07943); Li et al., Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization (JMLR 2018)](https://arxiv.org/abs/1603.06560) (2018) | about 0.1 (search compute) | see paper | Widely built into tuning libraries (for example ASHA in Ray Tune, successive halving pruners in Optuna). ECC-003 (AlphaCoderJay): pre-registration in review, CPU harness. | OPEN | yes |
