# Pre-registration: C27 successive halving vs random search (ECC-003)

Frozen at: the time of the merge commit that adds this file to `main` (METHOD: the commit time is the freeze time)
Author: AlphaCoderJay
Review: Noven (DotrixAI maintainer), two rounds on PR #1. Eight design choices proposed in the second review were confirmed by the author on 2026-10-27.

## Claim

- Claim id and text: C27 - Successive halving can find good hyperparameter configurations faster by a wide margin compared to random search under a comparable compute budget.
- Source and claimed ratio:  Jamieson & Talwalkar (2015), Non-stochastic Best Arm Identification and Hyperparameter Optimization; Li et al. (2018), Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.

- Claim being tested: Whether successive halving requires a lot less search compute than random search to find a configuration reaching a common specified target quality.

- Scope: This experiment tests the claim at the ECC baseline v1 scale. It does not establish that this result can be the same to other model sizes, datasets or search spaces. This is only for the ECC baseline v1.

## Setup

- Model and size: ECC baseline v1, a GPT styled Transformer Architecture with a width of 96,3 layers, 4 attention heads and MLP hidden dimension 256; approximately 0.73M parameters at a vocabulary size of 2,048. Search configs will vary model width and depth with the preregistered search space, subject to the harness parameter limit.
- Data and tokenizer: WikiText-2 (raw v1), prepared with `harness/dataprep/prepare_data.py`. Tokenizer: the custom 2,048-token byte-level BPE from ECC-001, `harness/dataprep/tokenizer.json` (SHA-256 ee98f08a2af1f3ef075394ca7f4197dfced91a34ed468cd891dc231c5ae5b6ed).
- Context length: 128.
- Batch size: 16 rows, as said in the harness
- Full training budget: 1,200 updates per configuration
- Baseline recipe: `harness/baselines/ecc_baseline_v1.py`. Tune the baseline learning rate on the grid {1e-3, 2e-3, 4e-3, 8e-3} (centred on ECC-001's best value, 2e-3), seed 3000, scored on the selection-validation windows; if the best value is at a grid edge, extend by one doubling on that side, at most twice. Use the tuned baseline's final score after 1,200 updates to define the target quality.
- Methods: Random search and successive halving over the same hyperparameter space. Account for actual training FLOPs, including partial runs, rather than assume equal updates mean equal compute.
- Hardware and threads: Record CPU model, thread count, operating system, Python/Pytorch version, and relevant hardware and software settings. Keep conditions as consistent as possible so we can have fair runs.
- Seeds: Baseline seed: 3000. Search-repeat seeds: 3003, 3004, and 3005, paired between the two methods. Final evaluation seeds: 3101, 3102, and 3103 for each selected configuration. Record any changes before the relevant experiment begins.
- Pilot runs before freezing (how many, what they showed): Record the number of pilots, their purpose, and what they showed. Pilots may verify that the harness runs correctly; do not use sealed evaluation results to select configurations 

### Search Space
-------------------------
 - Use the same space for the methods

 - Learning rate: 1e-3 - 3e-2
 - Model width: Choice from {64, 96, 128 , 160}
 - Depth: Choice from {2, 3 , 4 ,6}
 - Warmup updates: Choice from {0, 20, 50, 100}
 - Lr-Schedule: Choice from {cosine decay to 10% of the initial rate, constant with a linear cooldown to 10% over the last 20% of updates}.
 - Weight decay: Choice from {0, 0.05, 0.1}
 - AdamW beta2: Choice from {0.95, 0.99}
 - Parameter limit: 2M parameters per configuration.


### Search Protocol
----------------------------------
  #### Random search
  - Sample 16 configs from the search space.
  - Train each config for up to 1,200 updates.
  - Evaluate on the selection-validation at checkpoints.
  - Train the configs one after another in the order they were sampled (sampling uses the repeat seed).
  - Track cumulative training FLOPs until a config first reaches target quality Q, or until the search budget has been used up.
  - If no config reaches the target quality, record each search as a failure to reach the target in its budget.


  ### Successive halving
  - Start with 32 configs.
  - Use a reduction factor eta = 2.
  - Training budgets per rung: 150, 300, 600 and 1,200 updates.
  - At each rung, evaluate all surviving configs on the selection-validation windows and retain the top half, ranked preregistered quality metric
  - Record every configs, its score, cumulative FLOPs, and elimination decision
  - Count all compute spent on eliminated configs toward the search budget
  - Each rung restarts training from scratch at its budget (this is what makes the total 4 × 4,800 = 19,200 updates, equal to random search's 16 × 1,200). Within a rung, configs are trained in a fixed random order drawn from the repeat seed.
  - The intended budget is 16 baseline-equivalent full runs per search repeat

### Compute budget
- Expected total: 3 repeats × 2 methods × about 16 full-run equivalents, plus baseline tuning and 6 final runs: roughly 100+ full-run equivalents, about 20-40 CPU-hours on a laptop CPU (ESTIMATED from ECC-001, about 10 minutes per 0.73M run; 2M configs are slower).
- If the projected total exceeds 50 CPU-hours, stop and agree a reduction with the maintainers before continuing, and report it as a deviation.
    
## Target quality
- Target Q: The tuned ECC baseline v1's final bits-per-token score after 1,200 updates (seed 3000), **measured on the selection-validation windows 256-511**.
- Selection-validation data: WikiText-2 validation windows 256-511.
- Sealed evaluation data: WikiText-2 validation windows 0-255, used only for the final report of the selected configurations (final evaluation seeds 3101-3103), never for selection, stopping or Q.
- Metric direction: Lower bits per token is better.
- Eval protocol: Verify that search selection and stopping decisions cannot access the sealed windows. The same baseline define Q must be used for both methods. We want to ensure fair evaluation so we can see which method is better under fair settings.
- Target attainment: A config reaches Q when its selection-validation score meets or surpasses the predefined target at an evaluation checkpoint. Evaluate every 50 updates (rung boundaries 150/300/600/1,200 are checkpoints). FLOPs-to-Q for a search = cumulative training FLOPs of that search up to the checkpoint where its first config reaches Q.


## Prediction
- Expected R_FLOP: Predict a median R of 0.5, with a plausible range from 0.1-1.0. This is only my prediction not a result
- Expected R_time: Predict a median CPU-time ratio of 0.5, with a plausible range of 0.1-1.0. The time results are secondary as background load can interrupt comparisons. Which would make it not a fair comparison
- Probability of meeting the order-of-magnitude claim: 15% before running the experiment
- Additional hypothesis: Successive halving has at least a 60% probability of achieving at least 20% lower FLOPs-to-target than random search. This is my prediction, not a result

## Pass rule
Use the median paired FLOPs ratio R as the primary decision statistic.

- HOLDS: Median R ≤ 0.1, consistent with an order-of-magnitude reduction in search compute.

- PARTIAL: 0.1 < median R ≤ 0.8, showing an improvement of at least 20% but not an order-of-magnitude reduction.

- DOES NOT HOLD: Median R > 0.8 when both methods reach Q in all three paired repeats.

- INCONCLUSIVE: Either method fails to reach Q in one or more repeats, invalid runs compromise the comparison, compute budgets differ materially without a defensible correction, or the observed repeats are too inconsistent for a reliable verdict.


## Stop rules

- A run is ended early if: The harness reports an invalid run, causality or a finite-loss check fails, the configs exceed the set model parameter budget, or successive halving eliminates the configuration
- What an early stop means for the verdict: Invalid runs must be investigated and reported; do not silently omit their compute. A configuration eliminated by successive halving counts toward the search budget but is not treated as a completed 1,200-update run. If a selected configuration never reaches Q, report a bound rather than extrapolating beyond observed evaluation points. If these issues prevent a fair comparison, report INCONCLUSIVE.
