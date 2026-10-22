# Pre-registration: C__ <short title>

Frozen at: To be recorded when the preregistration is committed
Author: AlphaCoderJay

## Claim

- Claim id and text: ECC-003 - Under an equal total search-compute budget, successive halving has al least a 60% probability of achieving at least 20% lower FLOPs-to-target than random search
- Source and claimed ratio: Research hypothesis proposed for ECC-003. The clamed margin is 20% lower FLOPs-to-target for successive halving relative to random search

## Setup

- Model and size: ECC baseline v1, a GPT styled Transformer Architecture with a width of 96,3 layers, 4 attention heads and MLP hidden dimension 256; approximately 0.73M parameters at a vocabulary size of 2,048. Search configs will vary model width and depth with the preregistered search space, subject to the harness parameter limit.
- Data and tokenizer: WikiText-2, tokenizer and exact data preperation to be verified from the harness.
- Context length, batch size, updates: Context length -> 128; full-run configurations use 1,200 updates. Batch size to be verified from the harness.
- Baseline recipe (normally harness/baselines/ecc_baseline_v1.py) and how it was tuned:harness/baselines/ecc_baseline_v1.py . Tune the baseline learning rate using the same search grid and tuning effort as the method under test.
- Method recipe and how it was tuned (same effort as the baseline): Random search and successive halving over the preregistered training-config and search space, with equal total search FLOPs for both methods. The search budget is initially set to the equivalent of 16 full baseline runs
- Hardware and threads: Record CPU model, thread count, and relevant software versions before running. Run comparisons under similar/consistet conditions
- Seeds: Record seeds before baseline tuning, search runs, and final returns. Keep the final evaluation seeds separate from the search proccess where i'm able to
- Pilot runs before freezing (how many, what they showed): Record the number of pilots, their purpose, and what they showed. Pilots may verify that the harness runs correctly; do not use sealed evaluation results to select configurations 

## Target quality

- How Q is chosen (from the baseline, near the end of its budget): Q is the ECC baselines v1's final bits per token score after 1,200 updates. Use the same baseline reference run to define Q for both search methods
- Evaluation text: Final evaluation uses the sealed WikiText-2 validation windows 0–255. Search configuration selection must use separate validation windows 256–511. The evaluation pipeline must be verified to enforce this separation before the preregistration is frozen.

## Prediction

- Expected R_FLOP at Q, with a range: To be specified before the experiment. Report the expected FLOPs-to-target ratio relative to the baseline for each method.
- Expected R_time at Q, with a range: To be specified before the experiment. Report the corresponding CPU-time ratio relative to the baseline.
- Probability that the pass rule is met: 60% probability that successive halving achieves at least 20% lower FLOPs-to-target than random search under the stated conditions. This is a prediction, not a measured result.

## Pass rule

- The claim holds at this scale if: The selected selected successive-halving config requires no more than 80% of the FLOPs-to-target required by the selected random search config, using the same baseline-defined target quality Q and the same evaluation protocol
- Report both methods' total search FLOPs separately from the selected configurations' FLOPs-to-target. Report CPU-time results separately; a FLOPs improvement alone does not establish a time improvement.
- Rerun the best configuration from each method using three seeds per configuration as a separate final-evaluation budget. Report the individual seed results and their aggregate. If the evidence is too noisy or either method fails to reach Q, report the relevant bounds and assess whether the verdict is inconclusive.

## Stop rules

- A run is ended early if: The harness reports an invalid run, causality or a finite-loss check fails, the configs exceed the set model parameter budget, or successive halving eliminates the configuration
- What an early stop means for the verdict: nvalid runs must be investigated and reported; do not silently omit their compute. A configuration eliminated by successive halving counts toward the search budget but is not treated as a completed 1,200-update run. If a selected configuration never reaches Q, report a bound rather than extrapolating beyond observed evaluation points. If these issues prevent a fair comparison, report INCONCLUSIVE.
